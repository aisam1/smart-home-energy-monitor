import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:io';
import 'package:universal_html/html.dart' as html;
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

void main() {
  runApp(const EnergyApp());
}

class EnergyApp extends StatelessWidget {
  const EnergyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Smart Energy Monitor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1565C0),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const MainScreen(),
    );
  }
}

const String projectId = 'energy-monitor-5ecb7';
const String firestoreBase =
    'https://firestore.googleapis.com/v1/projects/$projectId'
    '/databases/(default)/documents';

// ── Time-of-use tariff helpers ────────────────────────────────────────────────
double getTariff(int hour, int dayOfWeek) {
  // dayOfWeek: DateTime.weekday — 1=Mon, 7=Sun
  final isWeekend = dayOfWeek >= 6;
  if (isWeekend) return 0.12;
  if (hour >= 7 && hour < 22) return 0.20;
  return 0.10;
}

String getTariffPeriod(int hour, int dayOfWeek) {
  final isWeekend = dayOfWeek >= 6;
  if (isWeekend) return 'Weekend';
  if (hour >= 7 && hour < 22) return 'Peak';
  return 'Off-Peak';
}

Color getTariffColor(String period) {
  switch (period) {
    case 'Peak':     return Colors.red;
    case 'Off-Peak': return Colors.green;
    case 'Weekend':  return Colors.orange;
    default:         return Colors.lightBlue;
  }
}

// ── Models ────────────────────────────────────────────────────────────────────
class EnergyReading {
  final double totalPower, fridge, washingMachine, tv,
               lighting, waterHeater, oven, estimatedBill;
  final String timestamp;
  final double currentTariff;
  final String tariffPeriod;
  final Map<String, bool> applianceStates;

  EnergyReading({
    required this.totalPower, required this.fridge,
    required this.washingMachine, required this.tv,
    required this.lighting, required this.waterHeater,
    required this.oven, required this.estimatedBill,
    required this.timestamp, required this.applianceStates,
    this.currentTariff = 0.15,
    this.tariffPeriod  = 'Unknown',
  });

  factory EnergyReading.fromFirestore(Map<String, dynamic> f) {
    Map<String, bool> states = {};
    final statesField = f['appliance_states'];
    if (statesField?['mapValue']?['fields'] != null) {
      final fields = statesField['mapValue']['fields']
          as Map<String, dynamic>;
      fields.forEach((k, v) {
        states[k] = v['booleanValue'] ?? true;
      });
    }
    return EnergyReading(
      totalPower:      _d(f['total_power']),
      fridge:          _d(f['fridge']),
      washingMachine:  _d(f['washing_machine']),
      tv:              _d(f['tv']),
      lighting:        _d(f['lighting']),
      waterHeater:     _d(f['water_heater']),
      oven:            _d(f['oven']),
      estimatedBill:   _d(f['estimated_bill_so_far']),
      timestamp:       f['timestamp']?['stringValue'] ?? '',
      currentTariff:   _d(f['current_tariff']) > 0
                           ? _d(f['current_tariff']) : 0.15,
      tariffPeriod:    f['tariff_period']?['stringValue'] ?? 'Unknown',
      applianceStates: states,
    );
  }

  static double _d(dynamic f) {
    if (f == null) return 0.0;
    if (f['doubleValue']  != null)
      return (f['doubleValue'] as num).toDouble();
    if (f['integerValue'] != null)
      return double.parse(f['integerValue'].toString());
    return 0.0;
  }
}

class ApplianceAlert {
  final String appliance, severity, message, timestamp;
  final double zscore;

  ApplianceAlert({
    required this.appliance, required this.severity,
    required this.message, required this.timestamp,
    required this.zscore,
  });

  factory ApplianceAlert.fromFirestore(Map<String, dynamic> f) =>
    ApplianceAlert(
      appliance: f['appliance']?['stringValue'] ?? '',
      severity:  f['severity']?['stringValue']  ?? 'warning',
      message:   f['message']?['stringValue']   ?? '',
      timestamp: f['timestamp']?['stringValue'] ?? '',
      zscore:    EnergyReading._d(f['zscore']),
    );

  String get severityLabel {
    if (zscore > 20) return 'Extremely abnormal';
    if (zscore > 10) return 'Very abnormal';
    if (zscore > 5)  return 'Abnormal';
    return 'Slightly abnormal';
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────
class MainScreen extends StatefulWidget {
  const MainScreen({super.key});
  @override State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _idx = 0;
  EnergyReading?       _latest;
  List<EnergyReading>  _history = [];
  List<ApplianceAlert> _alerts  = [];
  Map<String, dynamic> _ml      = {};
  Map<String, dynamic> _lstm    = {};
  bool                 _loading = true;
  Timer?               _timer;

  @override
  void initState() {
    super.initState();
    _fetchAll();
    _timer = Timer.periodic(
      const Duration(minutes: 2), (_) => _fetchAll());
  }

  @override
  void dispose() { _timer?.cancel(); super.dispose(); }

  Future<void> _fetchAll() async {
    await Future.wait([_fetchReadings(), _fetchAlerts(),
                       _fetchML(), _fetchLSTM()]);
    if (mounted) setState(() => _loading = false);
  }

  Future<void> _fetchReadings() async {
    try {
      final r = await http.get(Uri.parse(
        '$firestoreBase/energy_readings'
        '?orderBy=timestamp%20desc&pageSize=30'));
      if (r.statusCode == 200) {
        final docs = json.decode(r.body)['documents'] as List? ?? [];
        final list = docs.map((d) => EnergyReading.fromFirestore(
            d['fields'] as Map<String, dynamic>)).toList();
        if (mounted) setState(() {
          _history = list;
          _latest  = list.isNotEmpty ? list.first : null;
        });
      }
    } catch (_) {}
  }

  Future<void> _fetchAlerts() async {
    try {
      final r = await http.get(Uri.parse(
        '$firestoreBase/appliance_alerts?pageSize=30'));
      if (r.statusCode == 200) {
        final docs = json.decode(r.body)['documents'] as List? ?? [];
        final list = docs.map((d) => ApplianceAlert.fromFirestore(
            d['fields'] as Map<String, dynamic>)).toList();
        list.sort((a, b) => b.zscore.compareTo(a.zscore));
        if (mounted) setState(() => _alerts = list);
      }
    } catch (_) {}
  }

  Future<void> _fetchML() async {
    try {
      final r = await http.get(Uri.parse(
        '$firestoreBase/ml_results/latest'));
      if (r.statusCode == 200) {
        if (mounted) setState(() =>
          _ml = json.decode(r.body)['fields']
              as Map<String, dynamic>);
      }
    } catch (_) {}
  }

  Future<void> _fetchLSTM() async {
    try {
      final r = await http.get(Uri.parse(
        '$firestoreBase/lstm_predictions'
        '?orderBy=timestamp%20desc&pageSize=1'));
      if (r.statusCode == 200) {
        final docs = json.decode(r.body)['documents'] as List? ?? [];
        if (docs.isNotEmpty && mounted) setState(() =>
          _lstm = docs.first['fields'] as Map<String, dynamic>);
      }
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    final screens = [
      DashboardScreen(latest: _latest, history: _history,
                      alerts: _alerts, loading: _loading),
      AppliancesScreen(latest: _latest, history: _history),
      AlertsScreen(alerts: _alerts),
      HistoryScreen(history: _history),
      MLInsightsScreen(ml: _ml, lstm: _lstm, latest: _latest),
    ];

    return Scaffold(
      body: _loading && _latest == null
          ? const Center(child: CircularProgressIndicator())
          : IndexedStack(index: _idx, children: screens),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _idx,
        onDestinationSelected: (i) => setState(() => _idx = i),
        backgroundColor: const Color(0xFF1A1F2E),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.dashboard),     label: 'Dashboard'),
          NavigationDestination(
            icon: Icon(Icons.kitchen),       label: 'Appliances'),
          NavigationDestination(
            icon: Icon(Icons.warning_amber), label: 'Alerts'),
          NavigationDestination(
            icon: Icon(Icons.history),       label: 'History'),
          NavigationDestination(
            icon: Icon(Icons.psychology),    label: 'ML Insights'),
        ],
      ),
    );
  }
}

// ── Screen 1: Dashboard ───────────────────────────────────────────────────────
class DashboardScreen extends StatelessWidget {
  final EnergyReading?       latest;
  final List<EnergyReading>  history;
  final List<ApplianceAlert> alerts;
  final bool                 loading;

  const DashboardScreen({super.key, required this.latest,
    required this.history, required this.alerts,
    required this.loading});

  @override
  Widget build(BuildContext context) {
    final critical =
        alerts.where((a) => a.severity == 'critical').length;

    // Current tariff from live data or calculated locally
    final now    = DateTime.now();
    final tariff = (latest?.currentTariff ?? 0) > 0
    ? latest!.currentTariff
    : getTariff(now.hour, now.weekday);
final period = (latest?.tariffPeriod ?? 'Unknown') != 'Unknown'
    ? latest!.tariffPeriod
    : getTariffPeriod(now.hour, now.weekday);
    final tariffColor = getTariffColor(period);

    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1565C0),
        title: const Row(children: [
          Icon(Icons.bolt, color: Colors.amber),
          SizedBox(width: 8),
          Text('Smart Energy Monitor',
            style: TextStyle(color: Colors.white,
                             fontWeight: FontWeight.bold)),
        ]),
      ),
      body: latest == null
          ? const Center(child: Text(
              'No data — run live_simulation.py',
              style: TextStyle(color: Colors.grey)))
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (critical > 0)
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(12),
                      margin: const EdgeInsets.only(bottom: 12),
                      decoration: BoxDecoration(
                        color: Colors.red.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: Colors.red.withOpacity(0.5))),
                      child: Row(children: [
                        const Icon(Icons.warning, color: Colors.red),
                        const SizedBox(width: 8),
                        Expanded(child: Text(
                          '$critical critical anomalies — check Alerts tab',
                          style: const TextStyle(color: Colors.red,
                            fontWeight: FontWeight.bold))),
                      ]),
                    ),

                  // ── Tariff period banner ──────────────────────────────────
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 10),
                    margin: const EdgeInsets.only(bottom: 12),
                    decoration: BoxDecoration(
                      color: tariffColor.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                        color: tariffColor.withOpacity(0.4))),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Row(children: [
                          Icon(Icons.electric_bolt,
                            color: tariffColor, size: 16),
                          const SizedBox(width: 6),
                          Text('Current tariff: $period',
                            style: TextStyle(
                              color: tariffColor,
                              fontWeight: FontWeight.bold,
                              fontSize: 13)),
                        ]),
                        Text('€${tariff.toStringAsFixed(2)}/kWh',
                          style: TextStyle(
                            color: tariffColor,
                            fontWeight: FontWeight.bold,
                            fontSize: 13)),
                      ],
                    ),
                  ),

                  Row(children: [
                    Container(width: 10, height: 10,
                      decoration: const BoxDecoration(
                        color: Colors.green,
                        shape: BoxShape.circle)),
                    const SizedBox(width: 8),
                    const Text('Live Data', style: TextStyle(
                      color: Colors.green,
                      fontWeight: FontWeight.bold)),
                    const Spacer(),
                    Text(
                      latest!.timestamp.length > 19
                          ? latest!.timestamp.substring(0, 19)
                          : latest!.timestamp,
                      style: const TextStyle(
                        color: Colors.grey, fontSize: 12)),
                  ]),
                  const SizedBox(height: 12),

                  _PowerGauge(power: latest!.totalPower),
                  const SizedBox(height: 12),

                  Row(children: [
                    Expanded(child: _metricCard('This Week',
                      '€${latest!.estimatedBill.toStringAsFixed(2)}',
                      Icons.date_range, Colors.amber)),
                    const SizedBox(width: 8),
                    Expanded(child: _metricCard('Monthly',
                      '€${(latest!.estimatedBill * 4.33).toStringAsFixed(2)}',
                      Icons.calendar_month, Colors.lightBlue)),
                    const SizedBox(width: 8),
                    Expanded(child: _metricCard('Yearly',
                      '€${(latest!.estimatedBill * 52).toStringAsFixed(0)}',
                      Icons.calendar_today, Colors.green)),
                  ]),
                  const SizedBox(height: 12),

                  // ── Tariff reference ──────────────────────────────────────
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: const Color(0xFF1A1F2E),
                      borderRadius: BorderRadius.circular(10)),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _tariffBadge('🔴 Peak',
                          '€0.20/kWh', '07-22 weekdays',
                          Colors.red),
                        _tariffBadge('🟢 Off-peak',
                          '€0.10/kWh', '22-07 weekdays',
                          Colors.green),
                        _tariffBadge('🟡 Weekend',
                          '€0.12/kWh', 'All day',
                          Colors.orange),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),

                  const Text('Power Trend',
                    style: TextStyle(color: Colors.white,
                      fontSize: 16, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  _LineChart(readings: history.take(20)
                      .toList().reversed.toList()),
                  const SizedBox(height: 16),

                  const Text('Current Appliance Breakdown',
                    style: TextStyle(color: Colors.white,
                      fontSize: 16, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  _ApplianceCostTable(
                    reading: latest!,
                    tariff: tariff),
                  const SizedBox(height: 16),

                  const Text('Recent Readings',
                    style: TextStyle(color: Colors.white,
                      fontSize: 16, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  ...history.take(5).map((r) => _readingRow(r)),
                ],
              ),
            ),
    );
  }

  Widget _tariffBadge(String label, String rate,
                       String hours, Color color) =>
    Column(children: [
      Text(label, style: TextStyle(
        color: color, fontSize: 11,
        fontWeight: FontWeight.bold)),
      Text(rate, style: TextStyle(
        color: color, fontSize: 12,
        fontWeight: FontWeight.bold)),
      Text(hours, style: const TextStyle(
        color: Colors.grey, fontSize: 9)),
    ]);

  Widget _readingRow(EnergyReading r) {
    final ts  = DateTime.tryParse(r.timestamp);
    final tar = ts != null
        ? getTariff(ts.hour, ts.weekday) : 0.15;
    final per = ts != null
        ? getTariffPeriod(ts.hour, ts.weekday) : '';
    final col = getTariffColor(per);
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(
        horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(8)),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(r.timestamp.length > 19
              ? r.timestamp.substring(11, 19) : r.timestamp,
            style: const TextStyle(
              color: Colors.grey, fontSize: 13)),
          Text('${r.totalPower.toStringAsFixed(2)} kW',
            style: const TextStyle(color: Colors.lightBlue,
                                    fontWeight: FontWeight.bold)),
          Container(
            padding: const EdgeInsets.symmetric(
              horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: col.withOpacity(0.15),
              borderRadius: BorderRadius.circular(4)),
            child: Text('€${tar.toStringAsFixed(2)}',
              style: TextStyle(
                color: col, fontSize: 11,
                fontWeight: FontWeight.bold))),
          Text('€${r.estimatedBill.toStringAsFixed(2)}',
            style: const TextStyle(color: Colors.amber)),
        ],
      ),
    );
  }

  Widget _metricCard(String label, String value,
                      IconData icon, Color color) =>
    Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3))),
      child: Column(children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(height: 6),
        Text(value, style: TextStyle(color: color, fontSize: 15,
                                      fontWeight: FontWeight.bold)),
        Text(label, style: const TextStyle(
          color: Colors.grey, fontSize: 11)),
      ]),
    );
}

// ── Appliance Cost Table ──────────────────────────────────────────────────────
class _ApplianceCostTable extends StatelessWidget {
  final EnergyReading reading;
  final double        tariff;
  const _ApplianceCostTable({
    required this.reading,
    this.tariff = 0.15});

  @override
  Widget build(BuildContext context) {
    final appliances = [
      {'name': 'Fridge',          'power': reading.fridge,
       'color': Colors.lightBlue, 'icon': Icons.kitchen},
      {'name': 'Water Heater',    'power': reading.waterHeater,
       'color': Colors.orange,    'icon': Icons.water_drop},
      {'name': 'Washing Machine', 'power': reading.washingMachine,
       'color': Colors.purple,    'icon': Icons.local_laundry_service},
      {'name': 'Oven',            'power': reading.oven,
       'color': Colors.red,       'icon': Icons.microwave},
      {'name': 'TV',              'power': reading.tv,
       'color': Colors.teal,      'icon': Icons.tv},
      {'name': 'Lighting',        'power': reading.lighting,
       'color': Colors.yellow,    'icon': Icons.light},
    ];

    final totalPower = appliances
        .map((a) => a['power'] as double)
        .reduce((a, b) => a + b);

    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12)),
      child: Column(children: [
        Padding(
          padding: const EdgeInsets.all(12),
          child: Row(children: const [
            Expanded(flex: 3, child: Text('Appliance',
              style: TextStyle(color: Colors.grey, fontSize: 11))),
            Expanded(flex: 2, child: Text('Power',
              style: TextStyle(color: Colors.grey, fontSize: 11),
              textAlign: TextAlign.center)),
            Expanded(flex: 2, child: Text('Cost/hr',
              style: TextStyle(color: Colors.grey, fontSize: 11),
              textAlign: TextAlign.center)),
            Expanded(flex: 2, child: Text('Daily Est.',
              style: TextStyle(color: Colors.grey, fontSize: 11),
              textAlign: TextAlign.right)),
          ]),
        ),
        const Divider(color: Color(0xFF2A3148), height: 1),
        ...appliances.map((app) {
          final power   = app['power'] as double;
          final color   = app['color'] as Color;
          final isOn    = power > 0;
          // Use live TOU tariff for current cost
          final costHr  = power * tariff;
          // Daily estimate uses weighted avg tariff
          // (15h peak + 9h off-peak on weekdays)
          final avgDayTariff = 0.15;
          final costDay = power * 24 * avgDayTariff;
          final pct     = totalPower > 0
              ? (power / totalPower * 100) : 0.0;
          return Container(
            padding: const EdgeInsets.symmetric(
              horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              border: Border(bottom: BorderSide(
                color: const Color(0xFF2A3148), width: 0.5))),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Icon(app['icon'] as IconData,
                    color: isOn ? color : Colors.grey, size: 16),
                  const SizedBox(width: 6),
                  Expanded(flex: 3, child: Text(
                    app['name'] as String,
                    style: TextStyle(
                      color: isOn ? Colors.white : Colors.grey,
                      fontSize: 12, fontWeight: FontWeight.bold))),
                  Expanded(flex: 2, child: Text(
                    isOn ? '${power.toStringAsFixed(2)} kW' : 'OFF',
                    style: TextStyle(
                      color: isOn ? color : Colors.grey,
                      fontSize: 12),
                    textAlign: TextAlign.center)),
                  Expanded(flex: 2, child: Text(
                    isOn ? '€${costHr.toStringAsFixed(3)}' : '—',
                    style: const TextStyle(
                      color: Colors.white70, fontSize: 12),
                    textAlign: TextAlign.center)),
                  Expanded(flex: 2, child: Text(
                    isOn ? '€${costDay.toStringAsFixed(2)}' : '—',
                    style: TextStyle(
                      color: isOn ? Colors.amber : Colors.grey,
                      fontSize: 12, fontWeight: FontWeight.bold),
                    textAlign: TextAlign.right)),
                ]),
                if (isOn) ...[
                  const SizedBox(height: 4),
                  Row(children: [
                    const SizedBox(width: 22),
                    Expanded(child: ClipRRect(
                      borderRadius: BorderRadius.circular(2),
                      child: LinearProgressIndicator(
                        value: pct / 100,
                        backgroundColor: color.withOpacity(0.1),
                        valueColor: AlwaysStoppedAnimation<Color>(
                          color.withOpacity(0.6)),
                        minHeight: 3,
                      ),
                    )),
                    const SizedBox(width: 8),
                    Text('${pct.toStringAsFixed(0)}%',
                      style: TextStyle(color: color, fontSize: 10)),
                  ]),
                ],
              ],
            ),
          );
        }),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.03),
            borderRadius: const BorderRadius.only(
              bottomLeft: Radius.circular(12),
              bottomRight: Radius.circular(12))),
          child: Row(children: [
            const Expanded(flex: 3, child: Text('TOTAL',
              style: TextStyle(color: Colors.white,
                fontSize: 12, fontWeight: FontWeight.bold))),
            Expanded(flex: 2, child: Text(
              '${totalPower.toStringAsFixed(2)} kW',
              style: const TextStyle(color: Colors.lightBlue,
                fontSize: 12, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center)),
            Expanded(flex: 2, child: Text(
              '€${(totalPower * tariff).toStringAsFixed(3)}',
              style: const TextStyle(
                color: Colors.white70, fontSize: 12),
              textAlign: TextAlign.center)),
            Expanded(flex: 2, child: Text(
              '€${(totalPower * 24 * 0.15).toStringAsFixed(2)}',
              style: const TextStyle(color: Colors.amber,
                fontSize: 12, fontWeight: FontWeight.bold),
              textAlign: TextAlign.right)),
          ]),
        ),
      ]),
    );
  }
}

// ── Animated Power Gauge ──────────────────────────────────────────────────────
class _PowerGauge extends StatefulWidget {
  final double power;
  const _PowerGauge({required this.power});
  @override State<_PowerGauge> createState() => _PowerGaugeState();
}

class _PowerGaugeState extends State<_PowerGauge>
    with SingleTickerProviderStateMixin {
  late AnimationController _ctrl;
  late Animation<double>   _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(vsync: this,
        duration: const Duration(milliseconds: 1200));
    _anim = Tween<double>(begin: 0, end: widget.power)
        .animate(CurvedAnimation(
          parent: _ctrl, curve: Curves.easeOut));
    _ctrl.forward();
  }

  @override
  void didUpdateWidget(_PowerGauge old) {
    super.didUpdateWidget(old);
    if (old.power != widget.power) {
      _anim = Tween<double>(begin: old.power, end: widget.power)
          .animate(CurvedAnimation(
            parent: _ctrl, curve: Curves.easeOut));
      _ctrl.forward(from: 0);
    }
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _anim,
      builder: (_, __) => Container(
        width: double.infinity,
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF1565C0), Color(0xFF0D47A1)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight),
          borderRadius: BorderRadius.circular(16)),
        child: Column(children: [
          const Icon(Icons.bolt, color: Colors.amber, size: 44),
          const SizedBox(height: 4),
          Text('${_anim.value.toStringAsFixed(2)} kW',
            style: const TextStyle(color: Colors.white,
              fontSize: 48, fontWeight: FontWeight.bold)),
          const Text('Current Power Usage',
            style: TextStyle(color: Colors.white70, fontSize: 15)),
          const SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: (_anim.value / 10).clamp(0.0, 1.0),
              backgroundColor: Colors.white24,
              valueColor: AlwaysStoppedAnimation<Color>(
                _anim.value > 5 ? Colors.red
                    : _anim.value > 3 ? Colors.orange
                    : Colors.green),
              minHeight: 8,
            ),
          ),
          const SizedBox(height: 4),
          Row(mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: const [
              Text('0 kW', style: TextStyle(
                color: Colors.white54, fontSize: 10)),
              Text('5 kW', style: TextStyle(
                color: Colors.white54, fontSize: 10)),
              Text('10 kW', style: TextStyle(
                color: Colors.white54, fontSize: 10)),
            ]),
        ]),
      ),
    );
  }
}

// ── Line Chart ────────────────────────────────────────────────────────────────
class _LineChart extends StatelessWidget {
  final List<EnergyReading> readings;
  const _LineChart({required this.readings});

  @override
  Widget build(BuildContext context) {
    if (readings.isEmpty) return const SizedBox();
    return Container(
      height: 120,
      padding: const EdgeInsets.fromLTRB(8, 8, 8, 24),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12)),
      child: CustomPaint(
        painter: _LinePainter(readings),
        size: Size.infinite,
      ),
    );
  }
}

class _LinePainter extends CustomPainter {
  final List<EnergyReading> readings;
  _LinePainter(this.readings);

  @override
  void paint(Canvas canvas, Size size) {
    if (readings.length < 2) return;
    final values = readings.map((r) => r.totalPower).toList();
    final minV   = values.reduce(math.min);
    final maxV   = values.reduce(math.max);
    final range  = (maxV - minV).clamp(0.1, double.infinity);

    final fillPath = Path();
    for (int i = 0; i < readings.length; i++) {
      final x = (i / (readings.length - 1)) * size.width;
      final y = size.height -
          ((values[i] - minV) / range) * size.height;
      if (i == 0) fillPath.moveTo(x, y);
      else fillPath.lineTo(x, y);
    }
    fillPath.lineTo(size.width, size.height);
    fillPath.lineTo(0, size.height);
    fillPath.close();
    canvas.drawPath(fillPath, Paint()
      ..shader = LinearGradient(colors: [
        Colors.lightBlue.withOpacity(0.3),
        Colors.lightBlue.withOpacity(0.02)],
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
      ).createShader(
        Rect.fromLTWH(0, 0, size.width, size.height)));

    final path = Path();
    for (int i = 0; i < readings.length; i++) {
      final x = (i / (readings.length - 1)) * size.width;
      final y = size.height -
          ((values[i] - minV) / range) * size.height;
      if (i == 0) path.moveTo(x, y);
      else path.lineTo(x, y);
    }
    canvas.drawPath(path, Paint()
      ..color = Colors.lightBlue
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke);

    final dotPaint = Paint()
      ..color = Colors.lightBlue
      ..style = PaintingStyle.fill;

    for (final i in [0, readings.length ~/ 2, readings.length - 1]) {
      final x = (i / (readings.length - 1)) * size.width;
      final y = size.height -
          ((values[i] - minV) / range) * size.height;
      canvas.drawCircle(Offset(x, y), 4, dotPaint);

      final tp = TextPainter(
        text: TextSpan(
          text: '${values[i].toStringAsFixed(1)} kW',
          style: const TextStyle(
            color: Colors.white70, fontSize: 9)),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(x - tp.width / 2, y - 16));

      final r   = readings[i];
      final lbl = r.timestamp.length > 19
          ? r.timestamp.substring(11, 16) : '';
      final tp2 = TextPainter(
        text: TextSpan(text: lbl,
          style: const TextStyle(
            color: Colors.grey, fontSize: 9)),
        textDirection: TextDirection.ltr,
      )..layout();
      tp2.paint(canvas, Offset(
        (x - tp2.width / 2).clamp(0, size.width - tp2.width),
        size.height + 4));
    }
  }

  @override
  bool shouldRepaint(_LinePainter old) => old.readings != readings;
}

// ── Screen 2: Appliances ──────────────────────────────────────────────────────
class AppliancesScreen extends StatefulWidget {
  final EnergyReading?      latest;
  final List<EnergyReading> history;
  const AppliancesScreen({super.key, required this.latest,
                           required this.history});
  @override State<AppliancesScreen> createState() =>
      _AppliancesScreenState();
}

class _AppliancesScreenState extends State<AppliancesScreen>
    with AutomaticKeepAliveClientMixin {

  final Map<String, double> _simulatedWatts = {
    'Fridge': 0.15, 'Water Heater': 3.0,
    'Washing Machine': 2.0, 'Oven': 2.0,
    'TV': 0.12, 'Lighting': 0.06,
  };

  final Map<String, bool> _manualToggle = {};
  bool _sendingCommand = false;

  static const Map<String, String> _keyMap = {
    'Fridge':          'fridge',
    'Water Heater':    'water_heater',
    'Washing Machine': 'washing_machine',
    'Oven':            'oven',
    'TV':              'tv',
    'Lighting':        'lighting',
  };

  @override
  void initState() {
    super.initState();
    _loadCurrentCommands();
  }

  @override
  bool get wantKeepAlive => true;

  Future<void> _loadCurrentCommands() async {
    try {
      final r = await http.get(Uri.parse(
        '$firestoreBase/device_commands/current'));
      if (r.statusCode == 200) {
        final fields = json.decode(r.body)['fields']
            as Map<String, dynamic>? ?? {};
        if (mounted) {
          setState(() {
            fields.forEach((k, v) {
              final name = _keyMap.entries
                  .firstWhere((e) => e.value == k,
                    orElse: () => const MapEntry('', ''))
                  .key;
              if (name.isNotEmpty) {
                _manualToggle[name] = v['booleanValue'] ?? true;
              }
            });
          });
        }
      }
    } catch (_) {}
  }

  Future<void> _sendCommand(String name, bool value) async {
    if (_sendingCommand) return;
    setState(() => _sendingCommand = true);
    final key = _keyMap[name] ?? name.toLowerCase();

    try {
      final getRes = await http.get(Uri.parse(
        '$firestoreBase/device_commands/current'));
      Map<String, bool> current = {
        'fridge': true, 'water_heater': true,
        'washing_machine': true, 'oven': true,
        'tv': true, 'lighting': true,
      };
      if (getRes.statusCode == 200) {
        final fields = json.decode(getRes.body)['fields']
            as Map<String, dynamic>? ?? {};
        fields.forEach((k, v) {
          current[k] = v['booleanValue'] ?? true;
        });
      }
      current[key] = value;
      final patchRes = await http.patch(
        Uri.parse('$firestoreBase/device_commands/current'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'fields': current.map(
          (k, v) => MapEntry(k, {'booleanValue': v}))}),
      );
      if (patchRes.statusCode == 200 && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('$name turned '
            '${value ? "ON ✓" : "OFF ✓"} — '
            'updates in ~50 seconds'),
          backgroundColor: value ? Colors.green : Colors.orange,
          duration: const Duration(seconds: 3)));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Error: $e'),
          backgroundColor: Colors.red));
      }
    }
    if (mounted) setState(() => _sendingCommand = false);
  }

  double _liveValue(String name, EnergyReading r) {
    switch (name) {
      case 'Fridge':          return r.fridge;
      case 'Water Heater':    return r.waterHeater;
      case 'Washing Machine': return r.washingMachine;
      case 'Oven':            return r.oven;
      case 'TV':              return r.tv;
      case 'Lighting':        return r.lighting;
      default:                return 0.0;
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    if (widget.latest == null) {
      return const Scaffold(
        backgroundColor: Color(0xFF0D1117),
        body: Center(child: Text('No data yet',
          style: TextStyle(color: Colors.grey))));
    }

    final now    = DateTime.now();
    final tariff = getTariff(now.hour, now.weekday);
    final period = getTariffPeriod(now.hour, now.weekday);
    final tariffColor = getTariffColor(period);

    final appliances = [
      {'name': 'Fridge',
       'icon': Icons.kitchen,               'color': Colors.lightBlue},
      {'name': 'Water Heater',
       'icon': Icons.water_drop,            'color': Colors.orange},
      {'name': 'Washing Machine',
       'icon': Icons.local_laundry_service, 'color': Colors.purple},
      {'name': 'Oven',
       'icon': Icons.microwave,             'color': Colors.red},
      {'name': 'TV',
       'icon': Icons.tv,                    'color': Colors.teal},
      {'name': 'Lighting',
       'icon': Icons.light,                 'color': Colors.yellow},
    ];

    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1565C0),
        title: const Text('Appliances',
          style: TextStyle(color: Colors.white,
                           fontWeight: FontWeight.bold)),
        actions: [
          if (_sendingCommand)
            const Padding(
              padding: EdgeInsets.all(16),
              child: SizedBox(width: 20, height: 20,
                child: CircularProgressIndicator(
                  strokeWidth: 2, color: Colors.white))),
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: _loadCurrentCommands,
            tooltip: 'Refresh appliance states',
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Tariff banner
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: tariffColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: tariffColor.withOpacity(0.3))),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(children: [
                    Icon(Icons.access_time,
                      color: tariffColor, size: 16),
                    const SizedBox(width: 8),
                    Text('$period rate now — '
                      '€${tariff.toStringAsFixed(2)}/kWh',
                      style: TextStyle(
                        color: tariffColor,
                        fontWeight: FontWeight.bold,
                        fontSize: 12)),
                  ]),
                  Text(
                    period == 'Peak'
                      ? 'Consider delaying heavy use'
                      : 'Good time to run appliances',
                    style: TextStyle(
                      color: tariffColor, fontSize: 11)),
                ],
              ),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: Colors.blue.withOpacity(0.3))),
              child: const Row(children: [
                Icon(Icons.info_outline,
                  color: Colors.lightBlue, size: 16),
                SizedBox(width: 8),
                Expanded(child: Text(
                  'Toggle sends command to simulation. '
                  'Power changes visible in ~50 seconds.',
                  style: TextStyle(
                    color: Colors.lightBlue, fontSize: 12))),
              ]),
            ),
            const SizedBox(height: 16),

            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate:
                const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 2, crossAxisSpacing: 12,
                  mainAxisSpacing: 12, childAspectRatio: 1.05),
              itemCount: appliances.length,
              itemBuilder: (context, index) {
                final app   = appliances[index];
                final name  = app['name'] as String;
                final color = app['color'] as Color;
                final isOn  = _manualToggle[name] ?? true;
                final live  = _liveValue(name, widget.latest!);
                final val   = isOn
                    ? (live > 0 ? live : _simulatedWatts[name]!)
                    : 0.0;
                // Use live TOU tariff for current cost
                final costNow = val * tariff;
                // Daily estimate
                final daily = val * 24 * 0.15;

                return GestureDetector(
                  onTap: () => _showDetail(
                    context, name, color,
                    app['icon'] as IconData, val, tariff),
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: isOn
                          ? color.withOpacity(0.15)
                          : const Color(0xFF1A1F2E),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: isOn
                          ? color.withOpacity(0.5)
                          : const Color(0xFF2A3148))),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(app['icon'] as IconData,
                          color: isOn ? color : Colors.grey,
                          size: 30),
                        const SizedBox(height: 5),
                        Text(name,
                          style: TextStyle(
                            color: isOn ? Colors.white : Colors.grey,
                            fontSize: 12,
                            fontWeight: FontWeight.bold),
                          textAlign: TextAlign.center),
                        Text(isOn
                            ? '${val.toStringAsFixed(2)} kW'
                            : 'OFF',
                          style: TextStyle(
                            color: isOn ? color : Colors.grey,
                            fontSize: 11)),
                        if (isOn)
                          Text('€${costNow.toStringAsFixed(3)}/hr',
                            style: TextStyle(
                              color: tariffColor,
                              fontSize: 10,
                              fontWeight: FontWeight.bold)),
                        Text(isOn
                            ? '€${daily.toStringAsFixed(2)}/day est.'
                            : '',
                          style: const TextStyle(
                            color: Colors.grey, fontSize: 9)),
                        const SizedBox(height: 4),
                        Transform.scale(
                          scale: 0.75,
                          child: Switch(
                            value: isOn,
                            onChanged: _sendingCommand
                                ? null
                                : (v) {
                                    setState(() =>
                                      _manualToggle[name] = v);
                                    _sendCommand(name, v);
                                  },
                            activeColor: color,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
            const SizedBox(height: 16),

            const Text('Cost Breakdown',
              style: TextStyle(color: Colors.white,
                fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            _ApplianceCostTable(
              reading: widget.latest!,
              tariff: tariff),
          ],
        ),
      ),
    );
  }

  void _showDetail(BuildContext context, String name,
                   Color color, IconData icon,
                   double currentVal, double tariff) {
    final values = widget.history
        .map((r) => _liveValue(name, r)).toList();
    final avg   = values.isEmpty ? 0.0
                : values.reduce((a, b) => a + b) / values.length;
    final peak  = values.isEmpty ? 0.0 : values.reduce(math.max);
    final daily = currentVal * 24 * 0.15;

    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF1A1F2E),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(20))),
      builder: (ctx) => Padding(
        padding: const EdgeInsets.all(24),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Row(children: [
            Icon(icon, color: color, size: 32),
            const SizedBox(width: 12),
            Text(name, style: TextStyle(color: color,
              fontSize: 22, fontWeight: FontWeight.bold)),
          ]),
          const SizedBox(height: 16),
          _dr('Current usage',   '${currentVal.toStringAsFixed(3)} kW'),
          _dr('Session avg',     '${avg.toStringAsFixed(3)} kW'),
          _dr('Session peak',    '${peak.toStringAsFixed(3)} kW'),
          _dr('Cost now (TOU)',  '€${(currentVal * tariff).toStringAsFixed(3)}/hr'),
          _dr('Est. daily',      '€${daily.toStringAsFixed(2)}'),
          _dr('Est. monthly',    '€${(daily * 30).toStringAsFixed(2)}'),
          _dr('Est. yearly',     '€${(daily * 365).toStringAsFixed(2)}'),
          const SizedBox(height: 12),
        ]),
      ),
    );
  }

  Widget _dr(String l, String v) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 7),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(l, style: const TextStyle(color: Colors.grey)),
        Text(v, style: const TextStyle(color: Colors.white,
                                        fontWeight: FontWeight.bold)),
      ]),
  );
}

// ── Screen 3: Alerts ──────────────────────────────────────────────────────────
class AlertsScreen extends StatelessWidget {
  final List<ApplianceAlert> alerts;
  const AlertsScreen({super.key, required this.alerts});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1565C0),
        title: Row(children: [
          const Text('Alerts', style: TextStyle(
            color: Colors.white, fontWeight: FontWeight.bold)),
          const SizedBox(width: 8),
          if (alerts.isNotEmpty)
            Container(
              padding: const EdgeInsets.symmetric(
                horizontal: 8, vertical: 2),
              decoration: BoxDecoration(color: Colors.red,
                borderRadius: BorderRadius.circular(12)),
              child: Text('${alerts.length}',
                style: const TextStyle(
                  color: Colors.white, fontSize: 12)),
            ),
        ]),
      ),
      body: alerts.isEmpty
          ? const Center(child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.check_circle,
                  color: Colors.green, size: 64),
                SizedBox(height: 16),
                Text('No alerts', style: TextStyle(
                  color: Colors.grey, fontSize: 18)),
              ]))
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: alerts.length,
              itemBuilder: (_, i) {
                final a     = alerts[i];
                final color = a.severity == 'critical'
                    ? Colors.red
                    : a.severity == 'high'
                        ? Colors.orange : Colors.yellow;
                final icon  = a.severity == 'critical'
                    ? Icons.error
                    : a.severity == 'high'
                        ? Icons.warning : Icons.info;

                return Container(
                  margin: const EdgeInsets.only(bottom: 12),
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.08),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: color.withOpacity(0.4))),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(children: [
                        Icon(icon, color: color, size: 20),
                        const SizedBox(width: 8),
                        Expanded(child: Text(a.appliance,
                          style: TextStyle(color: color,
                            fontWeight: FontWeight.bold))),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 3),
                          decoration: BoxDecoration(
                            color: color.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(8)),
                          child: Text(a.severity.toUpperCase(),
                            style: TextStyle(color: color,
                              fontSize: 10,
                              fontWeight: FontWeight.bold)),
                        ),
                      ]),
                      const SizedBox(height: 8),
                      Text(a.message, style: const TextStyle(
                        color: Colors.white70, fontSize: 13)),
                      const SizedBox(height: 6),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: color.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(6)),
                        child: Text('⚡ ${a.severityLabel}',
                          style: TextStyle(
                            color: color, fontSize: 11)),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        a.timestamp.length > 16
                            ? a.timestamp.substring(0, 16)
                            : a.timestamp,
                        style: const TextStyle(
                          color: Colors.grey, fontSize: 11)),
                    ],
                  ),
                );
              }),
    );
  }
}

// ── Screen 4: History ─────────────────────────────────────────────────────────
class HistoryScreen extends StatefulWidget {
  final List<EnergyReading> history;
  const HistoryScreen({super.key, required this.history});
  @override State<HistoryScreen> createState() =>
      _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen>
    with AutomaticKeepAliveClientMixin {

  DateTime? _from, _to;
  int       _visibleCount = 10;

  @override
  bool get wantKeepAlive => true;

  List<EnergyReading> get _filtered {
    if (_from == null && _to == null) return widget.history;
    return widget.history.where((r) {
      if (r.timestamp.length < 10) return true;
      try {
        final d = DateTime.parse(r.timestamp.substring(0, 10));
        if (_from != null && d.isBefore(_from!)) return false;
        if (_to   != null && d.isAfter(_to!))   return false;
        return true;
      } catch (_) { return true; }
    }).toList();
  }

  Future<void> _pickFrom(BuildContext ctx) async {
    final d = await showDatePicker(context: ctx,
      initialDate: _from ?? DateTime.now(),
      firstDate: DateTime(2006), lastDate: DateTime.now());
    if (d != null) setState(() => _from = d);
  }

  Future<void> _pickTo(BuildContext ctx) async {
    final d = await showDatePicker(context: ctx,
      initialDate: _to ?? DateTime.now(),
      firstDate: DateTime(2006), lastDate: DateTime.now());
    if (d != null) setState(() => _to = d);
  }

  Future<void> _export() async {
    final f   = _filtered;
    final csv = [
      'timestamp,power_kw,fridge_kw,water_heater_kw,'
      'washing_machine_kw,oven_kw,tv_kw,lighting_kw,'
      'bill_eur,tariff_eur_kwh,tariff_period',
      ...f.map((r) {
        final ts  = DateTime.tryParse(r.timestamp);
        final tar = ts != null
            ? getTariff(ts.hour, ts.weekday) : 0.15;
        final per = ts != null
            ? getTariffPeriod(ts.hour, ts.weekday) : '';
        return '${r.timestamp},${r.totalPower},${r.fridge},'
            '${r.waterHeater},${r.washingMachine},${r.oven},'
            '${r.tv},${r.lighting},${r.estimatedBill},'
            '$tar,$per';
      })
    ].join('\n');

    if (kIsWeb) {
      final bytes  = utf8.encode(csv);
      final blob   = html.Blob([bytes], 'text/csv');
      final url    = html.Url.createObjectUrlFromBlob(blob);
      final anchor = html.AnchorElement(href: url)
        ..setAttribute('download', 'energy_export.csv')
        ..click();
      html.Url.revokeObjectUrl(url);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('CSV downloaded!'),
          backgroundColor: Colors.green));
      }
    } else {
      try {
        final dir  = await getTemporaryDirectory();
        final file = File('${dir.path}/energy_export.csv');
        await file.writeAsString(csv);
        await Share.shareXFiles(
          [XFile(file.path)],
          subject: 'Energy Monitor Export',
          text: 'Energy readings — ${f.length} rows');
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(
            content: Text('Export failed: $e'),
            backgroundColor: Colors.red));
        }
      }
    }
  }

  Map<String, Map<String, double>> _getWeeklyStats() {
    final f = _filtered;
    final Map<String, List<double>> byWeek = {};
    for (final r in f) {
      if (r.timestamp.length < 10) continue;
      try {
        final d    = DateTime.parse(r.timestamp.substring(0, 10));
        final week = 'W${_weekNumber(d)} ${d.year}';
        byWeek.putIfAbsent(week, () => []).add(r.totalPower);
      } catch (_) {}
    }
    return byWeek.map((k, v) => MapEntry(k, {
      'avg':   v.reduce((a, b) => a + b) / v.length,
      'total': v.reduce((a, b) => a + b),
      'count': v.length.toDouble(),
    }));
  }

  int _weekNumber(DateTime date) {
    final startOfYear = DateTime(date.year, 1, 1);
    final diff = date.difference(startOfYear).inDays;
    return (diff / 7).ceil() + 1;
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final f       = _filtered;
    final visible = f.take(_visibleCount).toList();

    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1565C0),
        title: const Text('History', style: TextStyle(
          color: Colors.white, fontWeight: FontWeight.bold)),
      ),
      body: widget.history.isEmpty
          ? const Center(child: Text('No history yet',
              style: TextStyle(color: Colors.grey)))
          : SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Filter by Date Range',
                    style: TextStyle(color: Colors.white,
                      fontSize: 16, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 10),
                  Row(children: [
                    Expanded(child: _datePicker(context, 'From',
                      _from, () => _pickFrom(context))),
                    const SizedBox(width: 10),
                    Expanded(child: _datePicker(context, 'To',
                      _to, () => _pickTo(context))),
                    if (_from != null || _to != null) ...[
                      const SizedBox(width: 8),
                      IconButton(
                        icon: const Icon(Icons.clear,
                          color: Colors.grey),
                        onPressed: () => setState(() {
                          _from = null; _to = null;
                        })),
                    ],
                  ]),
                  const SizedBox(height: 12),

                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: _export,
                      icon: const Icon(Icons.download),
                      label: Text(
                        'Export ${f.length} readings to CSV'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF1565C0),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(
                          vertical: 12)),
                    ),
                  ),
                  const SizedBox(height: 16),

                  if (f.isNotEmpty) ...[
                    const Text('Summary',
                      style: TextStyle(color: Colors.white,
                        fontSize: 16, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 10),
                    Row(children: [
                      Expanded(child: _statCard('Readings',
                        '${f.length}', Colors.lightBlue)),
                      const SizedBox(width: 10),
                      Expanded(child: _statCard('Avg Power',
                        '${(f.map((r) => r.totalPower).reduce((a, b) => a + b) / f.length).toStringAsFixed(2)} kW',
                        Colors.orange)),
                      const SizedBox(width: 10),
                      Expanded(child: _statCard('Latest Bill',
                        '€${f.first.estimatedBill.toStringAsFixed(2)}',
                        Colors.amber)),
                    ]),
                    const SizedBox(height: 10),
                    Row(children: [
                      Expanded(child: _statCard('Min Power',
                        '${f.map((r) => r.totalPower).reduce(math.min).toStringAsFixed(2)} kW',
                        Colors.green)),
                      const SizedBox(width: 10),
                      Expanded(child: _statCard('Max Power',
                        '${f.map((r) => r.totalPower).reduce(math.max).toStringAsFixed(2)} kW',
                        Colors.red)),
                      const SizedBox(width: 10),
                      Expanded(child: _statCard('Total Cost',
                        '€${f.map((r) => r.estimatedBill).reduce(math.max).toStringAsFixed(2)}',
                        Colors.purple)),
                    ]),
                    const SizedBox(height: 16),

                    const Text('Power Consumption',
                      style: TextStyle(color: Colors.white,
                        fontSize: 16, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    _LineChart(readings: f.reversed.toList()),
                    const SizedBox(height: 16),

                    const Text('Weekly Summary',
                      style: TextStyle(color: Colors.white,
                        fontSize: 16, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    _buildWeeklySummary(),
                    const SizedBox(height: 16),

                    _buildWeekComparison(),
                    const SizedBox(height: 16),

                    _buildPeakHoursAnalysis(),
                    const SizedBox(height: 16),

                    _buildWhatIfCalculator(),
                    const SizedBox(height: 16),

                    _buildEnergySavingTips(),
                    const SizedBox(height: 16),
                  ],

                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('All Readings',
                        style: TextStyle(color: Colors.white,
                          fontSize: 16, fontWeight: FontWeight.bold)),
                      Text('${visible.length} of ${f.length}',
                        style: const TextStyle(
                          color: Colors.grey, fontSize: 12)),
                    ],
                  ),
                  const SizedBox(height: 8),
                  ...visible.map((r) => _buildReadingCard(r)),

                  if (_visibleCount < f.length)
                    Padding(
                      padding: const EdgeInsets.symmetric(
                        vertical: 12),
                      child: SizedBox(
                        width: double.infinity,
                        child: OutlinedButton.icon(
                          onPressed: () => setState(() =>
                            _visibleCount += 10),
                          icon: const Icon(Icons.expand_more,
                            color: Colors.lightBlue),
                          label: Text(
                            'Load more '
                            '(${f.length - _visibleCount} remaining)',
                            style: const TextStyle(
                              color: Colors.lightBlue)),
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(
                              color: Colors.lightBlue),
                            padding: const EdgeInsets.symmetric(
                              vertical: 12)),
                        ),
                      ),
                    ),

                  if (_visibleCount >= f.length && f.length > 10)
                    Center(
                      child: TextButton(
                        onPressed: () =>
                          setState(() => _visibleCount = 10),
                        child: const Text('Show less',
                          style: TextStyle(color: Colors.grey)),
                      ),
                    ),
                ],
              ),
            ),
    );
  }

  Widget _buildReadingCard(EnergyReading r) {
    final active = <String>[];
    if (r.fridge > 0)         active.add('Fridge');
    if (r.waterHeater > 0)    active.add('Heater');
    if (r.washingMachine > 0) active.add('Washer');
    if (r.oven > 0)           active.add('Oven');
    if (r.tv > 0)             active.add('TV');
    if (r.lighting > 0)       active.add('Light');

    final ts     = DateTime.tryParse(r.timestamp);
    final period = ts != null
        ? getTariffPeriod(ts.hour, ts.weekday) : '';
    final tColor = getTariffColor(period);

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(8)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(r.timestamp.length > 19
                  ? r.timestamp.substring(0, 19) : r.timestamp,
                style: const TextStyle(
                  color: Colors.grey, fontSize: 11)),
              Text('${r.totalPower.toStringAsFixed(2)} kW',
                style: const TextStyle(color: Colors.lightBlue,
                  fontWeight: FontWeight.bold)),
              if (period.isNotEmpty)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 5, vertical: 1),
                  decoration: BoxDecoration(
                    color: tColor.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(4)),
                  child: Text(period,
                    style: TextStyle(
                      color: tColor, fontSize: 9,
                      fontWeight: FontWeight.bold))),
              Text('€${r.estimatedBill.toStringAsFixed(2)}',
                style: const TextStyle(color: Colors.amber)),
            ],
          ),
          if (active.isNotEmpty) ...[
            const SizedBox(height: 6),
            Wrap(spacing: 4, children: active
                .map((a) => Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: Colors.lightBlue.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(4)),
                  child: Text(a, style: const TextStyle(
                    color: Colors.lightBlue, fontSize: 9))))
                .toList()),
          ],
        ],
      ),
    );
  }

  Widget _buildWeeklySummary() {
    final weekly = _getWeeklyStats();
    if (weekly.isEmpty) {
      return const Text('Not enough data for weekly summary',
        style: TextStyle(color: Colors.grey));
    }
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12)),
      child: Column(
        children: weekly.entries.take(4).map((e) {
          final avg   = e.value['avg']!;
          final total = e.value['total']!;
          final cost  = total * (1/60) * 0.15;
          return Padding(
            padding: const EdgeInsets.all(12),
            child: Row(children: [
              Expanded(child: Text(e.key,
                style: const TextStyle(color: Colors.white,
                  fontSize: 13, fontWeight: FontWeight.bold))),
              Text('Avg: ${avg.toStringAsFixed(2)} kW',
                style: const TextStyle(
                  color: Colors.grey, fontSize: 12)),
              const SizedBox(width: 12),
              Text('€${cost.toStringAsFixed(2)}',
                style: const TextStyle(color: Colors.amber,
                  fontWeight: FontWeight.bold)),
            ]),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildWeekComparison() {
    final f = _filtered;
    if (f.length < 20) return const SizedBox();
    final half      = f.length ~/ 2;
    final recent    = f.take(half).toList();
    final older     = f.skip(half).toList();
    final recentAvg = recent.map((r) => r.totalPower)
        .reduce((a, b) => a + b) / recent.length;
    final olderAvg  = older.map((r) => r.totalPower)
        .reduce((a, b) => a + b) / older.length;
    final diff      = recentAvg - olderAvg;
    final pct       = olderAvg > 0
        ? (diff / olderAvg * 100).abs() : 0.0;
    final improved  = diff < 0;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: (improved ? Colors.green : Colors.orange)
              .withOpacity(0.3))),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Period Comparison',
            style: TextStyle(color: Colors.white,
              fontSize: 15, fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          Row(children: [
            Expanded(child: _compCard('Earlier Period',
              '${olderAvg.toStringAsFixed(2)} kW avg',
              Colors.grey)),
            const SizedBox(width: 12),
            Expanded(child: _compCard('Recent Period',
              '${recentAvg.toStringAsFixed(2)} kW avg',
              improved ? Colors.green : Colors.orange)),
          ]),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: (improved ? Colors.green : Colors.orange)
                  .withOpacity(0.1),
              borderRadius: BorderRadius.circular(8)),
            child: Row(children: [
              Icon(improved
                  ? Icons.trending_down : Icons.trending_up,
                color: improved ? Colors.green : Colors.orange),
              const SizedBox(width: 8),
              Expanded(child: Text(
                improved
                    ? 'Usage decreased by ${pct.toStringAsFixed(1)}% — great improvement!'
                    : 'Usage increased by ${pct.toStringAsFixed(1)}% — consider reducing consumption.',
                style: TextStyle(
                  color: improved ? Colors.green : Colors.orange,
                  fontSize: 13))),
            ]),
          ),
        ],
      ),
    );
  }

  Widget _compCard(String label, String value, Color color) =>
    Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8)),
      child: Column(children: [
        Text(value, style: TextStyle(color: color,
          fontSize: 14, fontWeight: FontWeight.bold)),
        Text(label, style: const TextStyle(
          color: Colors.grey, fontSize: 11)),
      ]),
    );

  Widget _buildPeakHoursAnalysis() {
    final f = _filtered;
    if (f.isEmpty) return const SizedBox();
    final Map<int, List<double>> byHour = {};
    for (final r in f) {
      if (r.timestamp.length < 19) continue;
      try {
        final hour = int.parse(r.timestamp.substring(11, 13));
        byHour.putIfAbsent(hour, () => []).add(r.totalPower);
      } catch (_) {}
    }
    if (byHour.isEmpty) return const SizedBox();
    final hourlyAvg = byHour.map((h, vals) =>
      MapEntry(h, vals.reduce((a, b) => a + b) / vals.length));
    final sorted   = hourlyAvg.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final topHours = sorted.take(3).toList();
    final lowHours = sorted.reversed.take(3).toList();

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Peak Hours Analysis',
            style: TextStyle(color: Colors.white,
              fontSize: 15, fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          const Text('🔴 Highest consumption hours:',
            style: TextStyle(color: Colors.red, fontSize: 13)),
          const SizedBox(height: 6),
          ...topHours.map((e) {
            final tariff = getTariff(e.key, 1); // weekday estimate
            final period = getTariffPeriod(e.key, 1);
            final tColor = getTariffColor(period);
            return Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Row(children: [
                const SizedBox(width: 8),
                Text('${e.key.toString().padLeft(2, '0')}:00',
                  style: const TextStyle(color: Colors.white,
                    fontSize: 13, fontWeight: FontWeight.bold)),
                const SizedBox(width: 12),
                Text('${e.value.toStringAsFixed(2)} kW avg',
                  style: const TextStyle(
                    color: Colors.grey, fontSize: 12)),
                const SizedBox(width: 8),
                Text('€${(e.value * tariff).toStringAsFixed(3)}/hr',
                  style: TextStyle(
                    color: tColor, fontSize: 12)),
                const SizedBox(width: 4),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 4, vertical: 1),
                  decoration: BoxDecoration(
                    color: tColor.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(3)),
                  child: Text(period,
                    style: TextStyle(
                      color: tColor, fontSize: 9))),
              ]),
            );
          }),
          const SizedBox(height: 12),
          const Text('🟢 Lowest consumption hours:',
            style: TextStyle(color: Colors.green, fontSize: 13)),
          const SizedBox(height: 6),
          ...lowHours.map((e) {
            final tariff = getTariff(e.key, 1);
            final period = getTariffPeriod(e.key, 1);
            final tColor = getTariffColor(period);
            return Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Row(children: [
                const SizedBox(width: 8),
                Text('${e.key.toString().padLeft(2, '0')}:00',
                  style: const TextStyle(color: Colors.white,
                    fontSize: 13, fontWeight: FontWeight.bold)),
                const SizedBox(width: 12),
                Text('${e.value.toStringAsFixed(2)} kW avg',
                  style: const TextStyle(
                    color: Colors.grey, fontSize: 12)),
                const SizedBox(width: 8),
                Text('€${(e.value * tariff).toStringAsFixed(3)}/hr',
                  style: TextStyle(
                    color: tColor, fontSize: 12)),
                const SizedBox(width: 4),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 4, vertical: 1),
                  decoration: BoxDecoration(
                    color: tColor.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(3)),
                  child: Text(period,
                    style: TextStyle(
                      color: tColor, fontSize: 9))),
              ]),
            );
          }),
          const SizedBox(height: 8),
          if (topHours.isNotEmpty && lowHours.isNotEmpty)
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8)),
              child: Text(
                '💡 Tip: Shift heavy appliances from '
                '${topHours.first.key}:00 '
                '(${getTariffPeriod(topHours.first.key, 1)} '
                '€${getTariff(topHours.first.key, 1).toStringAsFixed(2)}/kWh) to '
                '${lowHours.first.key}:00 '
                '(${getTariffPeriod(lowHours.first.key, 1)} '
                '€${getTariff(lowHours.first.key, 1).toStringAsFixed(2)}/kWh)',
                style: const TextStyle(
                  color: Colors.lightBlue, fontSize: 12)),
            ),
        ],
      ),
    );
  }

  Widget _buildWhatIfCalculator() {
    final f = _filtered;
    if (f.isEmpty) return const SizedBox();
    final avgPower    = f.map((r) => r.totalPower)
        .reduce((a, b) => a + b) / f.length;
    // Use weighted avg tariff for monthly estimate
    final monthlyCost = avgPower * 24 * 0.15 * 30;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: Colors.amber.withOpacity(0.3))),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(children: [
            Icon(Icons.calculate, color: Colors.amber),
            SizedBox(width: 8),
            Text('What-If Calculator',
              style: TextStyle(color: Colors.white,
                fontSize: 15, fontWeight: FontWeight.bold)),
          ]),
          const SizedBox(height: 12),
          const Text('Based on current usage + TOU tariff:',
            style: TextStyle(color: Colors.grey, fontSize: 12)),
          const SizedBox(height: 10),
          _whatIfRow(
            'Shift water heater to off-peak (22-07)',
            monthlyCost * 0.12, Colors.orange),
          _whatIfRow('Switch to LED lighting',
            monthlyCost * 0.05, Colors.yellow),
          _whatIfRow(
            'Run washing machine off-peak (saves peak rate)',
            monthlyCost * 0.09, Colors.purple),
          _whatIfRow('Reduce oven usage by 30min/day',
            monthlyCost * 0.04, Colors.red),
          const Divider(color: Color(0xFF2A3148)),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Total potential savings:',
                style: TextStyle(color: Colors.white,
                  fontWeight: FontWeight.bold)),
              Text(
                '€${(monthlyCost * 0.30).toStringAsFixed(2)}/month',
                style: const TextStyle(color: Colors.green,
                  fontSize: 16, fontWeight: FontWeight.bold)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _whatIfRow(String label, double saving, Color color) =>
    Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(child: Row(children: [
            Container(width: 8, height: 8,
              decoration: BoxDecoration(
                color: color, shape: BoxShape.circle)),
            const SizedBox(width: 8),
            Expanded(child: Text(label,
              style: const TextStyle(
                color: Colors.grey, fontSize: 12))),
          ])),
          Text('€${saving.toStringAsFixed(2)}/mo',
            style: TextStyle(color: color,
              fontWeight: FontWeight.bold, fontSize: 12)),
        ],
      ),
    );

  Widget _buildEnergySavingTips() {
    final f = _filtered;
    if (f.isEmpty) return const SizedBox();
    final avgPower = f.map((r) => r.totalPower)
        .reduce((a, b) => a + b) / f.length;
    final tips = <Map<String, dynamic>>[];
    if (avgPower > 2.0) {
      tips.add({
        'icon': Icons.warning_amber, 'color': Colors.orange,
        'title': 'High average consumption detected',
        'desc': 'Your average of ${avgPower.toStringAsFixed(2)} kW '
                'is above the typical 1.5 kW household average.',
      });
    }
    tips.addAll([
      {'icon': Icons.schedule, 'color': Colors.green,
       'title': 'Off-peak scheduling (€0.10/kWh)',
       'desc': 'Schedule washing machine and water heater '
               'between 22:00 and 07:00 on weekdays for the '
               'lowest rate. Half the cost of peak hours.'},
      {'icon': Icons.weekend, 'color': Colors.orange,
       'title': 'Weekend rate (€0.12/kWh)',
       'desc': 'Weekend consumption is billed at a flat €0.12/kWh '
               'all day — cheaper than weekday peak hours for '
               'running heavy appliances.'},
      {'icon': Icons.thermostat, 'color': Colors.red,
       'title': 'Avoid peak hours for water heater',
       'desc': 'Running a 3 kW water heater for 1 hour costs '
               '€0.60 at peak rate vs €0.30 off-peak. '
               'Timer scheduling saves ~€9/month.'},
      {'icon': Icons.eco, 'color': Colors.lightBlue,
       'title': 'Standby power',
       'desc': 'Devices on standby consume 5-10% of total energy. '
               'Use smart power strips to eliminate standby waste.'},
    ]);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF1A1F2E),
        borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(children: [
            Icon(Icons.tips_and_updates, color: Colors.green),
            SizedBox(width: 8),
            Text('Energy Saving Recommendations',
              style: TextStyle(color: Colors.white,
                fontSize: 15, fontWeight: FontWeight.bold)),
          ]),
          const SizedBox(height: 12),
          ...tips.map((tip) => Container(
            margin: const EdgeInsets.only(bottom: 10),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: (tip['color'] as Color).withOpacity(0.08),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: (tip['color'] as Color).withOpacity(0.2))),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(tip['icon'] as IconData,
                  color: tip['color'] as Color, size: 20),
                const SizedBox(width: 10),
                Expanded(child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(tip['title'] as String,
                      style: TextStyle(
                        color: tip['color'] as Color,
                        fontWeight: FontWeight.bold,
                        fontSize: 13)),
                    const SizedBox(height: 4),
                    Text(tip['desc'] as String,
                      style: const TextStyle(
                        color: Colors.grey, fontSize: 12,
                        height: 1.4)),
                  ],
                )),
              ],
            ),
          )),
        ],
      ),
    );
  }

  Widget _datePicker(BuildContext ctx, String label,
                      DateTime? date, VoidCallback onTap) =>
    GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: const Color(0xFF1A1F2E),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: const Color(0xFF2A3148))),
        child: Row(children: [
          const Icon(Icons.calendar_today,
            color: Colors.lightBlue, size: 16),
          const SizedBox(width: 8),
          Text(
            date == null ? label
                : '${date.year}-'
                  '${date.month.toString().padLeft(2, '0')}-'
                  '${date.day.toString().padLeft(2, '0')}',
            style: TextStyle(
              color: date == null ? Colors.grey : Colors.white,
              fontSize: 13)),
        ]),
      ),
    );

  Widget _statCard(String label, String value, Color color) =>
    Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3))),
      child: Column(children: [
        Text(value, style: TextStyle(color: color,
          fontSize: 13, fontWeight: FontWeight.bold)),
        Text(label, style: const TextStyle(
          color: Colors.grey, fontSize: 11)),
      ]),
    );
}

// ── Screen 5: ML Insights ─────────────────────────────────────────────────────
class MLInsightsScreen extends StatefulWidget {
  final Map<String, dynamic> ml, lstm;
  final EnergyReading?       latest;

  const MLInsightsScreen({super.key, required this.ml,
    required this.lstm, required this.latest});

  @override
  State<MLInsightsScreen> createState() => _MLInsightsScreenState();
}

class _MLInsightsScreenState extends State<MLInsightsScreen> {
  final TextEditingController _budgetController =
      TextEditingController(text: '30');
  double _monthlyBudget = 30.0;

  static double _d(dynamic f) {
    if (f == null) return 0.0;
    if (f['doubleValue']  != null)
      return (f['doubleValue'] as num).toDouble();
    if (f['integerValue'] != null)
      return double.parse(f['integerValue'].toString());
    return 0.0;
  }
  static String _s(dynamic f) => f?['stringValue'] ?? '';

  @override
  void dispose() {
    _budgetController.dispose();
    super.dispose();
  }

  List<Map<String, dynamic>> _getApplianceAdvice(
      List<Map<String, dynamic>> hourlyPreds) {
    if (hourlyPreds.isEmpty) return [];
    final appliances = [
      {'name': 'Water Heater', 'icon': Icons.water_drop,
       'color': Colors.orange, 'kw': 3.0, 'duration': 1,
       'tip': 'Heat water during off-peak hours (€0.10/kWh)'},
      {'name': 'Washing Machine', 'icon': Icons.local_laundry_service,
       'color': Colors.purple, 'kw': 2.0, 'duration': 2,
       'tip': 'Run a full load during cheapest TOU window'},
      {'name': 'Oven', 'icon': Icons.microwave,
       'color': Colors.red, 'kw': 2.0, 'duration': 1,
       'tip': 'Cook during off-peak to save up to 50%'},
      {'name': 'TV', 'icon': Icons.tv,
       'color': Colors.teal, 'kw': 0.12, 'duration': 2,
       'tip': 'Low impact — schedule freely'},
    ];

    return appliances.map((app) {
      final duration = app['duration'] as int;
      final appKw    = app['kw'] as double;
      String bestTime  = '--';
      String worstTime = '--';
      double savings   = 0.0;

      if (hourlyPreds.length >= duration) {
        double minCost    = double.infinity;
        double maxCost    = 0.0;
        int    bestStart  = 0;
        int    worstStart = 0;

        for (int i = 0; i <= hourlyPreds.length - duration; i++) {
          double windowCost = 0;
          for (int j = 0; j < duration; j++) {
            final pw      = hourlyPreds[i+j]['power'] as double;
            // Use tariff from Firebase if available
            final tariff  = hourlyPreds[i+j]['tariff'] as double?
                ?? getTariff(
                    hourlyPreds[i+j]['hour'] as int,
                    DateTime.now().weekday);
            windowCost += pw * tariff;
          }
          if (windowCost < minCost) {
            minCost = windowCost; bestStart = i;
          }
          if (windowCost > maxCost) {
            maxCost = windowCost; worstStart = i;
          }
        }
        bestTime  = '${(hourlyPreds[bestStart]['hour'] as int).toString().padLeft(2,'0')}:00';
        worstTime = '${(hourlyPreds[worstStart]['hour'] as int).toString().padLeft(2,'0')}:00';
        savings   = (maxCost - minCost) * appKw;
      }

      return {
        ...app,
        'best_time':  bestTime,
        'worst_time': worstTime,
        'savings':    savings.abs(),
      };
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final weekly   = _d(widget.ml['weekly_total']);
    final monthly  = _d(widget.ml['monthly_total']);
    final mae      = _d(widget.lstm['mae']);
    final rmse     = _d(widget.lstm['rmse']);
    final kwh24    = _d(widget.lstm['total_predicted_kwh']);
    final cost24   = _d(widget.lstm['total_predicted_cost']);
    final power    = widget.latest?.totalPower ?? 0.0;
    final bill     = widget.latest?.estimatedBill ?? 0.0;

    // Current tariff
    final now         = DateTime.now();
    final currTariff  = getTariff(now.hour, now.weekday);
    final currPeriod  = getTariffPeriod(now.hour, now.weekday);
    final tariffColor = getTariffColor(currPeriod);

    List<Map<String, dynamic>> preds = [];
    final pf = widget.ml['predictions'];
    if (pf?['arrayValue'] != null) {
      final vals = pf['arrayValue']['values'] as List? ?? [];
      for (final v in vals) {
        final f = v['mapValue']?['fields'];
        if (f != null) preds.add({
          'day':          _s(f['day']),
          'bill':         _d(f['predicted_bill']),
          'tariff_label': f['tariff_label']?['stringValue'] ?? '',
        });
      }
    }

    List<Map<String, dynamic>> hourlyPreds = [];
    final hf = widget.lstm['predictions'];
    if (hf?['arrayValue'] != null) {
      final vals = hf['arrayValue']['values'] as List? ?? [];
      for (final v in vals) {
        final f = v['mapValue']?['fields'];
        if (f != null) {
          final hour   = _d(f['hour']).toInt();
          final tariff = _d(f['tariff']) > 0
              ? _d(f['tariff'])
              : getTariff(hour, now.weekday);
          final period = f['tariff_period']?['stringValue']
              ?? getTariffPeriod(hour, now.weekday);
          hourlyPreds.add({
            'hour':    hour,
            'power':   _d(f['predicted_power']),
            'cost':    _d(f['predicted_cost']),
            'tariff':  tariff,
            'period':  period,
          });
        }
      }
    }

    final co2Week  = kwh24 * 7 * 0.6;
    final bestDay  = preds.isNotEmpty
        ? preds.reduce((a, b) =>
            (a['bill'] as double) < (b['bill'] as double)
                ? a : b)['day'] : '—';
    final worstDay = preds.isNotEmpty
        ? preds.reduce((a, b) =>
            (a['bill'] as double) > (b['bill'] as double)
                ? a : b)['day'] : '—';

    final projectedMonthly = bill * 30 * 4.33;
    final budgetPct = _monthlyBudget > 0
        ? (projectedMonthly / _monthlyBudget * 100).clamp(0.0, 200.0)
        : 0.0;
    final overBudget = projectedMonthly > _monthlyBudget;

    double historicalAvgPower = 0.0;
    final hpf = widget.ml['hourly_profile'];
    if (hpf?['arrayValue'] != null) {
      final vals = hpf['arrayValue']['values'] as List? ?? [];
      for (final v in vals) {
        final f = v['mapValue']?['fields'];
        if (f != null) {
          final m   = _d(f['month']).toInt();
          final dow = _d(f['dow']).toInt();
          final hr  = _d(f['hour']).toInt();
          if (m == now.month &&
              dow == now.weekday - 1 &&
              hr == now.hour) {
            historicalAvgPower = _d(f['avg_power']);
            break;
          }
        }
      }
    }
    final histDiff        = historicalAvgPower > 0
        ? ((power - historicalAvgPower) / historicalAvgPower * 100)
        : 0.0;
    final isOverHistorical = histDiff > 20;

    final dayNames   = ['Monday','Tuesday','Wednesday',
                        'Thursday','Friday','Saturday','Sunday'];
    final monthNames = ['','Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'];
    final shortDays  = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
    final applianceAdvice = _getApplianceAdvice(hourlyPreds);

    final banner = Container(
      width: double.infinity,
      margin: const EdgeInsets.fromLTRB(16, 16, 16, 0),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: const LinearGradient(colors: [
          Color(0xFF6A1B9A), Color(0xFF4A148C)]),
        borderRadius: BorderRadius.circular(12)),
      child: Column(children: [
        const Text('ML Model Performance',
          style: TextStyle(color: Colors.white70, fontSize: 13)),
        const SizedBox(height: 12),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            _mlStat('RF\nMAE',
              '€${_d(widget.ml['model_mae']).toStringAsFixed(3)}/day',
              Colors.greenAccent),
            _mlStat('LSTM\nMAE',
              '${mae.toStringAsFixed(3)} kW',
              Colors.lightBlueAccent),
            _mlStat('LSTM\nRMSE',
              '${rmse.toStringAsFixed(3)} kW',
              Colors.purpleAccent),
          ],
        ),
        const SizedBox(height: 8),
        const Text(
          'RF MAE = avg daily bill error in € • '
          'LSTM MAE = avg hourly power error in kW',
          style: TextStyle(color: Colors.white38, fontSize: 10),
          textAlign: TextAlign.center),
        const SizedBox(height: 6),
        // Current tariff in banner
        Container(
          padding: const EdgeInsets.symmetric(
            horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            color: tariffColor.withOpacity(0.2),
            borderRadius: BorderRadius.circular(8)),
          child: Text(
            'Now: $currPeriod — €${currTariff.toStringAsFixed(2)}/kWh',
            style: TextStyle(
              color: tariffColor,
              fontWeight: FontWeight.bold,
              fontSize: 11))),
      ]),
    );

    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1565C0),
        title: const Text('ML Insights', style: TextStyle(
          color: Colors.white, fontWeight: FontWeight.bold)),
      ),
      body: DefaultTabController(
        length: 3,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            banner,
            const SizedBox(height: 12),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Container(
                decoration: BoxDecoration(
                  color: const Color(0xFF1A1F2E),
                  borderRadius: BorderRadius.circular(12)),
                child: const TabBar(
                  labelColor: Colors.white,
                  unselectedLabelColor: Colors.grey,
                  indicatorColor: Color(0xFF1565C0),
                  indicatorSize: TabBarIndicatorSize.tab,
                  tabs: [
                    Tab(text: '7-Day Forecast'),
                    Tab(text: '24h LSTM'),
                    Tab(text: '💡 Smart Tips'),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 8),

            Expanded(
              child: TabBarView(children: [

                // ── Tab 1: 7-Day Forecast ──────────────────────────────────
                SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  physics: const ClampingScrollPhysics(),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        '7-Day Bill Prediction (Random Forest)',
                        style: TextStyle(color: Colors.white,
                          fontSize: 15, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      const Text(
                        'Predicted daily cost using TOU tariff',
                        style: TextStyle(
                          color: Colors.grey, fontSize: 12)),
                      const SizedBox(height: 12),
                      if (preds.isEmpty)
                        const Text(
                          'Run ml_model.py for predictions',
                          style: TextStyle(color: Colors.grey))
                      else ...[
                        SizedBox(
                          height: 180,
                          child: Row(
                            crossAxisAlignment:
                              CrossAxisAlignment.end,
                            children: preds.map((p) {
                              final b       = p['bill'] as double;
                              final maxB    = preds
                                  .map((x) => x['bill'] as double)
                                  .reduce(math.max);
                              final h       = maxB > 0
                                  ? (b / maxB) * 110 : 10.0;
                              final isBest  = p['day'] == bestDay;
                              final isWorst = p['day'] == worstDay;
                              return Expanded(child: Column(
                                mainAxisAlignment:
                                  MainAxisAlignment.end,
                                children: [
                                  if (isBest)
                                    const Text('🟢',
                                      style: TextStyle(fontSize: 10))
                                  else if (isWorst)
                                    const Text('🔴',
                                      style: TextStyle(fontSize: 10))
                                  else
                                    const SizedBox(height: 14),
                                  Text(
                                    '€${b.toStringAsFixed(2)}',
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 10),
                                    textAlign: TextAlign.center),
                                  const SizedBox(height: 4),
                                  Container(
                                    height: h,
                                    margin: const EdgeInsets
                                      .symmetric(horizontal: 3),
                                    decoration: BoxDecoration(
                                      gradient: LinearGradient(
                                        colors: isBest
                                          ? [Colors.green.shade300,
                                             Colors.green.shade700]
                                          : isWorst
                                            ? [Colors.red.shade300,
                                               Colors.red.shade700]
                                            : [Colors.blue.shade300,
                                               Colors.blue.shade700],
                                        begin: Alignment.topCenter,
                                        end: Alignment.bottomCenter),
                                      borderRadius:
                                        BorderRadius.circular(4))),
                                  const SizedBox(height: 6),
                                  Text(p['day'] as String,
                                    style: const TextStyle(
                                      color: Colors.grey,
                                      fontSize: 11),
                                    textAlign: TextAlign.center),
                                ],
                              ));
                            }).toList(),
                          ),
                        ),
                        const SizedBox(height: 12),
                        Row(children: [
                          Expanded(child: _infoCard(
                            'Weekly Total',
                            '€${weekly.toStringAsFixed(2)}',
                            Icons.calendar_view_week,
                            Colors.green)),
                          const SizedBox(width: 10),
                          Expanded(child: _infoCard(
                            'Monthly Est.',
                            '€${monthly.toStringAsFixed(2)}',
                            Icons.calendar_month,
                            Colors.lightBlue)),
                          const SizedBox(width: 10),
                          Expanded(child: _infoCard(
                            'Best Day 🟢', bestDay,
                            Icons.star, Colors.amber)),
                        ]),
                        const SizedBox(height: 12),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.blue.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(
                              color: Colors.blue.withOpacity(0.3))),
                          child: Text(
                            '💡 Schedule water heater, washing machine '
                            'and oven on $bestDay. '
                            'Avoid heavy usage on $worstDay. '
                            'Use off-peak hours (22-07) for maximum savings.',
                            style: const TextStyle(
                              color: Colors.lightBlue,
                              fontSize: 12, height: 1.4))),
                      ],
                    ],
                  ),
                ),

                // ── Tab 2: LSTM 24h ────────────────────────────────────────
                SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  physics: const ClampingScrollPhysics(),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('LSTM 24h Hourly Forecast',
                        style: TextStyle(color: Colors.white,
                          fontSize: 15, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      Builder(builder: (context) {
                        final desc = widget.lstm[
                            'context_description']
                            ?['stringValue'] as String?;
                        return Text(
                          desc ?? 'Context-aware prediction '
                                  'for today\'s pattern',
                          style: const TextStyle(
                            color: Colors.grey, fontSize: 12));
                      }),
                      const SizedBox(height: 12),
                      Row(children: [
                        Expanded(child: _infoCard('24h Energy',
                          '${kwh24.toStringAsFixed(1)} kWh',
                          Icons.battery_charging_full,
                          Colors.purple)),
                        const SizedBox(width: 8),
                        Expanded(child: _infoCard('24h Cost (TOU)',
                          '€${cost24.toStringAsFixed(2)}',
                          Icons.euro, Colors.amber)),
                        const SizedBox(width: 8),
                        Expanded(child: _infoCard('Avg/hour',
                          '${(hourlyPreds.isNotEmpty ? hourlyPreds.map((h) => h["power"] as double).reduce((a,b) => a+b) / hourlyPreds.length : 0).toStringAsFixed(2)} kW',
                          Icons.speed, Colors.teal)),
                      ]),
                      const SizedBox(height: 12),
                      if (hourlyPreds.isEmpty)
                        Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: const Color(0xFF1A1F2E),
                            borderRadius: BorderRadius.circular(12)),
                          child: const Center(child: Text(
                            'Run lstm_model.py to generate',
                            style: TextStyle(color: Colors.grey))))
                      else ...[
                        Container(
                          height: 160,
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: const Color(0xFF1A1F2E),
                            borderRadius: BorderRadius.circular(12)),
                          child: CustomPaint(
                            painter: _LSTMChartPainter(hourlyPreds),
                            size: Size.infinite,
                          ),
                        ),
                        const SizedBox(height: 8),
                        // Tariff legend
                        Row(
                          mainAxisAlignment:
                            MainAxisAlignment.center,
                          children: const [
                            _TariffLegendItem(
                              color: Colors.red,
                              label: '🔴 Peak €0.20'),
                            SizedBox(width: 16),
                            _TariffLegendItem(
                              color: Colors.green,
                              label: '🟢 Off-peak €0.10'),
                            SizedBox(width: 16),
                            _TariffLegendItem(
                              color: Colors.orange,
                              label: '🟡 Weekend €0.12'),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Container(
                          decoration: BoxDecoration(
                            color: const Color(0xFF1A1F2E),
                            borderRadius: BorderRadius.circular(12)),
                          child: Column(children: [
                            Padding(
                              padding: const EdgeInsets.all(10),
                              child: Row(children: const [
                                Expanded(child: Text('Hour',
                                  style: TextStyle(
                                    color: Colors.grey, fontSize: 11))),
                                Expanded(child: Text('Power',
                                  style: TextStyle(
                                    color: Colors.grey, fontSize: 11),
                                  textAlign: TextAlign.center)),
                                Expanded(child: Text('Cost',
                                  style: TextStyle(
                                    color: Colors.grey, fontSize: 11),
                                  textAlign: TextAlign.center)),
                                Expanded(child: Text('Rate',
                                  style: TextStyle(
                                    color: Colors.grey, fontSize: 11),
                                  textAlign: TextAlign.right)),
                              ]),
                            ),
                            const Divider(
                              color: Color(0xFF2A3148), height: 1),
                            ...hourlyPreds.map((p) {
                              final pw     = p['power'] as double;
                              final period = p['period'] as String?
                                  ?? getTariffPeriod(
                                      p['hour'] as int,
                                      now.weekday);
                              final tColor = getTariffColor(period);
                              final advice = period == 'Peak'
                                  ? '🔴 Peak'
                                  : period == 'Off-Peak'
                                      ? '🟢 Off-peak'
                                      : '🟡 Weekend';
                              return Container(
                                color: tColor.withOpacity(0.04),
                                child: Padding(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 10, vertical: 6),
                                  child: Row(children: [
                                    Expanded(child: Text(
                                      '${(p['hour'] as int).toString().padLeft(2,'0')}:00',
                                      style: const TextStyle(
                                        color: Colors.white,
                                        fontSize: 12))),
                                    Expanded(child: Text(
                                      '${pw.toStringAsFixed(2)} kW',
                                      style: const TextStyle(
                                        color: Colors.lightBlue,
                                        fontSize: 12),
                                      textAlign: TextAlign.center)),
                                    Expanded(child: Text(
                                      '€${(p['cost'] as double).toStringAsFixed(4)}',
                                      style: const TextStyle(
                                        color: Colors.amber,
                                        fontSize: 12),
                                      textAlign: TextAlign.center)),
                                    Expanded(child: Text(
                                      advice,
                                      style: TextStyle(
                                        color: tColor,
                                        fontSize: 11,
                                        fontWeight: FontWeight.bold),
                                      textAlign: TextAlign.right)),
                                  ]),
                                ),
                              );
                            }),
                          ]),
                        ),
                      ],
                    ],
                  ),
                ),

                // ── Tab 3: Smart Tips ──────────────────────────────────────
                SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  physics: const ClampingScrollPhysics(),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [

                      // Budget tracker
                      const Text('💰 Monthly Budget Tracker',
                        style: TextStyle(color: Colors.white,
                          fontSize: 15, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 10),
                      TextField(
                        controller: _budgetController,
                        keyboardType: TextInputType.number,
                        style: const TextStyle(color: Colors.white),
                        decoration: InputDecoration(
                          labelText: 'Monthly budget (€)',
                          labelStyle: const TextStyle(
                            color: Colors.grey),
                          prefixText: '€ ',
                          prefixStyle: const TextStyle(
                            color: Colors.green),
                          filled: true,
                          fillColor: const Color(0xFF1A1F2E),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                            borderSide: BorderSide.none),
                        ),
                        onChanged: (v) {
                          final parsed = double.tryParse(v);
                          if (parsed != null) {
                            setState(() => _monthlyBudget = parsed);
                          }
                        },
                      ),
                      const SizedBox(height: 12),

                      Container(
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: (overBudget
                              ? Colors.red
                              : budgetPct > 80
                                  ? Colors.orange
                                  : Colors.green)
                            .withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: (overBudget
                                ? Colors.red
                                : budgetPct > 80
                                    ? Colors.orange
                                    : Colors.green)
                              .withOpacity(0.4))),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(children: [
                              Icon(
                                overBudget
                                  ? Icons.warning
                                  : budgetPct > 80
                                      ? Icons.info
                                      : Icons.check_circle,
                                color: overBudget
                                  ? Colors.red
                                  : budgetPct > 80
                                      ? Colors.orange
                                      : Colors.green,
                                size: 20),
                              const SizedBox(width: 8),
                              Expanded(child: Text(
                                overBudget
                                  ? '⚠️ Over budget — projected €${projectedMonthly.toStringAsFixed(2)}/month'
                                  : budgetPct > 80
                                      ? '⚡ Approaching budget — projected €${projectedMonthly.toStringAsFixed(2)}/month'
                                      : '✅ On track — projected €${projectedMonthly.toStringAsFixed(2)}/month',
                                style: TextStyle(
                                  color: overBudget
                                    ? Colors.red
                                    : budgetPct > 80
                                        ? Colors.orange
                                        : Colors.green,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 13))),
                            ]),
                            const SizedBox(height: 10),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(4),
                              child: LinearProgressIndicator(
                                value: (budgetPct / 100)
                                  .clamp(0.0, 1.0),
                                backgroundColor: Colors.white12,
                                valueColor:
                                  AlwaysStoppedAnimation<Color>(
                                    overBudget
                                      ? Colors.red
                                      : budgetPct > 80
                                          ? Colors.orange
                                          : Colors.green),
                                minHeight: 8,
                              ),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              '${budgetPct.toStringAsFixed(0)}% of €${_monthlyBudget.toStringAsFixed(0)} budget',
                              style: const TextStyle(
                                color: Colors.grey, fontSize: 11)),
                            if (overBudget) ...[
                              const SizedBox(height: 10),
                              const Divider(
                                color: Colors.red, height: 1),
                              const SizedBox(height: 8),
                              Text(
                                '💡 Shift water heater to off-peak '
                                '(saves ~€${(3.0 * 0.10 * 30).toStringAsFixed(0)}/month) '
                                'or reduce oven usage during peak hours.',
                                style: const TextStyle(
                                  color: Colors.red,
                                  fontSize: 12, height: 1.4)),
                            ],
                          ],
                        ),
                      ),
                      const SizedBox(height: 20),

                      // Right Now vs Historical Average
                      if (historicalAvgPower > 0 || power > 0) ...[
                        const Text(
                          '⚡ Right Now vs Historical Average',
                          style: TextStyle(color: Colors.white,
                            fontSize: 15,
                            fontWeight: FontWeight.bold)),
                        const SizedBox(height: 10),
                        Container(
                          padding: const EdgeInsets.all(14),
                          decoration: BoxDecoration(
                            color: (isOverHistorical
                                ? Colors.orange
                                : Colors.green)
                              .withOpacity(0.1),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(
                              color: (isOverHistorical
                                  ? Colors.orange
                                  : Colors.green)
                                .withOpacity(0.3))),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                mainAxisAlignment:
                                  MainAxisAlignment.spaceBetween,
                                children: [
                                  Column(
                                    crossAxisAlignment:
                                      CrossAxisAlignment.start,
                                    children: [
                                      const Text('Current',
                                        style: TextStyle(
                                          color: Colors.grey,
                                          fontSize: 11)),
                                      Text(
                                        '${power.toStringAsFixed(2)} kW',
                                        style: TextStyle(
                                          color: isOverHistorical
                                            ? Colors.orange
                                            : Colors.green,
                                          fontSize: 20,
                                          fontWeight:
                                            FontWeight.bold)),
                                    ],
                                  ),
                                  Column(
                                    crossAxisAlignment:
                                      CrossAxisAlignment.end,
                                    children: [
                                      Text(
                                        'Hist. avg '
                                        '(${shortDays[now.weekday-1]} '
                                        '${now.hour}:00)',
                                        style: const TextStyle(
                                          color: Colors.grey,
                                          fontSize: 11)),
                                      Text(
                                        historicalAvgPower > 0
                                          ? '${historicalAvgPower.toStringAsFixed(2)} kW'
                                          : 'No data',
                                        style: const TextStyle(
                                          color: Colors.white,
                                          fontSize: 20,
                                          fontWeight:
                                            FontWeight.bold)),
                                    ],
                                  ),
                                ],
                              ),
                              const SizedBox(height: 8),
                              // Show current cost at live tariff
                              Row(
                                mainAxisAlignment:
                                  MainAxisAlignment.spaceBetween,
                                children: [
                                  Text(
                                    'Cost now: €${(power * currTariff).toStringAsFixed(4)}/hr',
                                    style: TextStyle(
                                      color: tariffColor,
                                      fontSize: 12,
                                      fontWeight: FontWeight.bold)),
                                  Container(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 8, vertical: 3),
                                    decoration: BoxDecoration(
                                      color: tariffColor.withOpacity(0.15),
                                      borderRadius: BorderRadius.circular(6)),
                                    child: Text(
                                      '$currPeriod €${currTariff.toStringAsFixed(2)}/kWh',
                                      style: TextStyle(
                                        color: tariffColor,
                                        fontSize: 10,
                                        fontWeight: FontWeight.bold))),
                                ],
                              ),
                              const SizedBox(height: 8),
                              Text(
                                historicalAvgPower == 0
                                  ? 'ℹ️ Run ml_model.py to load historical averages.'
                                  : isOverHistorical
                                    ? '⚠️ Using ${histDiff.abs().toStringAsFixed(0)}% more '
                                      'than the historical average for '
                                      '${dayNames[now.weekday-1]}s in '
                                      '${monthNames[now.month]} at '
                                      '${now.hour}:00. '
                                      'Consider turning off water heater or oven.'
                                    : '✅ Consumption is within normal '
                                      'range based on 4 years of '
                                      '${dayNames[now.weekday-1]} '
                                      '${monthNames[now.month]} data.',
                                style: TextStyle(
                                  color: historicalAvgPower == 0
                                    ? Colors.grey
                                    : isOverHistorical
                                        ? Colors.orange
                                        : Colors.green,
                                  fontSize: 12, height: 1.4)),
                            ],
                          ),
                        ),
                        const SizedBox(height: 20),
                      ],

                      // Appliance scheduling
                      const Text(
                        '🕐 Appliance Scheduling Advice',
                        style: TextStyle(color: Colors.white,
                          fontSize: 15, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 4),
                      const Text(
                        'Based on today\'s LSTM forecast + TOU tariff',
                        style: TextStyle(
                          color: Colors.grey, fontSize: 12)),
                      const SizedBox(height: 10),

                      if (applianceAdvice.isEmpty)
                        const Text(
                          'Run lstm_model.py for scheduling advice',
                          style: TextStyle(color: Colors.grey))
                      else
                        ...applianceAdvice.map((app) {
                          final color   = app['color'] as Color;
                          final savings = app['savings'] as double;
                          final bTime   = app['best_time'] as String;
                          final wTime   = app['worst_time'] as String;

                          // Get tariff for best/worst time
                          String bPeriod = '';
                          String wPeriod = '';
                          if (bTime != '--') {
                            final bHour = int.tryParse(
                              bTime.split(':')[0]) ?? 0;
                            bPeriod = getTariffPeriod(
                              bHour, now.weekday);
                          }
                          if (wTime != '--') {
                            final wHour = int.tryParse(
                              wTime.split(':')[0]) ?? 0;
                            wPeriod = getTariffPeriod(
                              wHour, now.weekday);
                          }

                          return Container(
                            margin: const EdgeInsets.only(bottom: 10),
                            padding: const EdgeInsets.all(14),
                            decoration: BoxDecoration(
                              color: color.withOpacity(0.08),
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(
                                color: color.withOpacity(0.3))),
                            child: Column(
                              crossAxisAlignment:
                                CrossAxisAlignment.start,
                              children: [
                                Row(children: [
                                  Icon(app['icon'] as IconData,
                                    color: color, size: 20),
                                  const SizedBox(width: 8),
                                  Text(app['name'] as String,
                                    style: TextStyle(
                                      color: color,
                                      fontWeight: FontWeight.bold,
                                      fontSize: 14)),
                                  const Spacer(),
                                  Container(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: 8, vertical: 3),
                                    decoration: BoxDecoration(
                                      color: Colors.green
                                        .withOpacity(0.2),
                                      borderRadius:
                                        BorderRadius.circular(8)),
                                    child: Text(
                                      'Save €${savings.toStringAsFixed(2)}',
                                      style: const TextStyle(
                                        color: Colors.green,
                                        fontSize: 11,
                                        fontWeight: FontWeight.bold))),
                                ]),
                                const SizedBox(height: 10),
                                Row(children: [
                                  Expanded(child: Column(
                                    crossAxisAlignment:
                                      CrossAxisAlignment.start,
                                    children: [
                                      const Text('✅ Best time',
                                        style: TextStyle(
                                          color: Colors.green,
                                          fontSize: 11)),
                                      Text(bTime,
                                        style: const TextStyle(
                                          color: Colors.white,
                                          fontSize: 18,
                                          fontWeight: FontWeight.bold)),
                                      if (bPeriod.isNotEmpty)
                                        Text(bPeriod,
                                          style: TextStyle(
                                            color: getTariffColor(
                                              bPeriod),
                                            fontSize: 10)),
                                    ],
                                  )),
                                  Expanded(child: Column(
                                    crossAxisAlignment:
                                      CrossAxisAlignment.end,
                                    children: [
                                      const Text('❌ Avoid',
                                        style: TextStyle(
                                          color: Colors.red,
                                          fontSize: 11)),
                                      Text(wTime,
                                        style: const TextStyle(
                                          color: Colors.white,
                                          fontSize: 18,
                                          fontWeight: FontWeight.bold)),
                                      if (wPeriod.isNotEmpty)
                                        Text(wPeriod,
                                          style: TextStyle(
                                            color: getTariffColor(
                                              wPeriod),
                                            fontSize: 10)),
                                    ],
                                  )),
                                ]),
                                const SizedBox(height: 6),
                                Text(app['tip'] as String,
                                  style: const TextStyle(
                                    color: Colors.grey, fontSize: 11)),
                              ],
                            ),
                          );
                        }),

                      const SizedBox(height: 20),

                      // Weekly best/worst days
                      const Text(
                        '📅 This Week — Best & Worst Days',
                        style: TextStyle(color: Colors.white,
                          fontSize: 15, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 10),

                      if (preds.isEmpty)
                        const Text(
                          'Run ml_model.py for weekly advice',
                          style: TextStyle(color: Colors.grey))
                      else ...[
                        ...preds.map((p) {
                          final d         = p['day'] as String;
                          final b         = p['bill'] as double;
                          final tLabel    = p['tariff_label'] as String;
                          final isBest    = d == bestDay;
                          final isWorst   = d == worstDay;
                          final color     = isBest
                              ? Colors.green
                              : isWorst ? Colors.red : Colors.grey;
                          return Container(
                            margin: const EdgeInsets.only(bottom: 8),
                            padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 10),
                            decoration: BoxDecoration(
                              color: color.withOpacity(0.08),
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(
                                color: color.withOpacity(0.3))),
                            child: Row(children: [
                              Text(
                                isBest ? '🟢'
                                    : isWorst ? '🔴' : '🟡',
                                style: const TextStyle(fontSize: 16)),
                              const SizedBox(width: 10),
                              Expanded(child: Column(
                                crossAxisAlignment:
                                  CrossAxisAlignment.start,
                                children: [
                                  Text(d, style: TextStyle(
                                    color: color,
                                    fontWeight: FontWeight.bold,
                                    fontSize: 14)),
                                  Text(
                                    isBest
                                      ? 'Best day — run heavy appliances'
                                      : isWorst
                                          ? 'Most expensive — avoid heavy usage'
                                          : 'Average consumption expected',
                                    style: const TextStyle(
                                      color: Colors.grey,
                                      fontSize: 11)),
                                  if (tLabel.isNotEmpty)
                                    Text(tLabel,
                                      style: const TextStyle(
                                        color: Colors.grey,
                                        fontSize: 9)),
                                ],
                              )),
                              Text('€${b.toStringAsFixed(2)}',
                                style: TextStyle(
                                  color: color,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 15)),
                            ]),
                          );
                        }),
                        const SizedBox(height: 12),
                        Builder(builder: (context) {
                          final maxBill = preds
                              .map((p) => p['bill'] as double)
                              .reduce(math.max);
                          final minBill = preds
                              .map((p) => p['bill'] as double)
                              .reduce(math.min);
                          final saving =
                              (maxBill - minBill).toStringAsFixed(2);
                          return Container(
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.blue.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(
                                color: Colors.blue.withOpacity(0.3))),
                            child: Text(
                              '💡 Potential weekly saving: €$saving '
                              'by shifting heavy appliance use '
                              'from $worstDay to $bestDay. '
                              'Additional savings by using off-peak rate (22-07).',
                              style: const TextStyle(
                                color: Colors.lightBlue,
                                fontSize: 12, height: 1.4)));
                        }),
                      ],

                      // CO2
                      const SizedBox(height: 20),
                      const Text('🌱 CO₂ Footprint',
                        style: TextStyle(color: Colors.white,
                          fontSize: 15, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 10),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Colors.green.withOpacity(0.08),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: Colors.green.withOpacity(0.3))),
                        child: Column(children: [
                          const Icon(Icons.eco,
                            color: Colors.green, size: 32),
                          const SizedBox(height: 8),
                          Text(
                            '${co2Week.toStringAsFixed(1)} kg CO₂/week',
                            style: const TextStyle(
                              color: Colors.green,
                              fontSize: 24,
                              fontWeight: FontWeight.bold)),
                          const SizedBox(height: 4),
                          Text(
                            '${(co2Week * 52).toStringAsFixed(0)} kg/year  •  '
                            '${(co2Week * 4.33).toStringAsFixed(0)} kg/month',
                            style: const TextStyle(
                              color: Colors.grey, fontSize: 12)),
                          const SizedBox(height: 8),
                          const Text(
                            '0.493 kg CO₂ per kWh '
                            '(Bosnia grid, 2025)',
                            style: TextStyle(
                              color: Colors.grey, fontSize: 11)),
                        ]),
                      ),
                      const SizedBox(height: 20),
                    ],
                  ),
                ),

              ]),
            ),
          ],
        ),
      ),
    );
  }

  Widget _mlStat(String label, String value, Color color) =>
    Column(children: [
      Text(value, style: TextStyle(color: color,
        fontSize: 20, fontWeight: FontWeight.bold)),
      const SizedBox(height: 4),
      Text(label, style: const TextStyle(
        color: Colors.white60, fontSize: 10),
        textAlign: TextAlign.center),
    ]);

  Widget _infoCard(String label, String value,
                   IconData icon, Color color) =>
    Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withOpacity(0.3))),
      child: Column(children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(height: 6),
        Text(value, style: TextStyle(color: color,
          fontSize: 13, fontWeight: FontWeight.bold),
          textAlign: TextAlign.center),
        Text(label, style: const TextStyle(
          color: Colors.grey, fontSize: 10),
          textAlign: TextAlign.center),
      ]),
    );
}

// ── Tariff Legend Item ────────────────────────────────────────────────────────
class _TariffLegendItem extends StatelessWidget {
  final Color  color;
  final String label;
  const _TariffLegendItem({required this.color, required this.label});

  @override
  Widget build(BuildContext context) =>
    Row(mainAxisSize: MainAxisSize.min, children: [
      Container(width: 10, height: 10,
        decoration: BoxDecoration(
          color: color.withOpacity(0.6),
          shape: BoxShape.circle)),
      const SizedBox(width: 4),
      Text(label, style: TextStyle(
        color: color, fontSize: 10)),
    ]);
}

// ── LSTM Chart Painter ────────────────────────────────────────────────────────
class _LSTMChartPainter extends CustomPainter {
  final List<Map<String, dynamic>> data;
  _LSTMChartPainter(this.data);

  @override
  void paint(Canvas canvas, Size size) {
    if (data.length < 2) return;
    final values = data.map((d) => d['power'] as double).toList();
    final minV   = values.reduce(math.min);
    final maxV   = values.reduce(math.max);
    final range  = (maxV - minV).clamp(0.1, double.infinity);
    final h      = size.height - 20;

    // Draw tariff period backgrounds
    for (int i = 0; i < data.length; i++) {
      final x1      = (i / (data.length - 1)) * size.width;
      final x2      = i < data.length - 1
          ? ((i + 1) / (data.length - 1)) * size.width
          : size.width;
      final period  = data[i]['period'] as String? ?? '';
      final bgColor = period == 'Peak'
          ? const Color(0x15e74c3c)
          : period == 'Off-Peak'
              ? const Color(0x152ecc71)
              : const Color(0x15f39c12);
      canvas.drawRect(
        Rect.fromLTWH(x1, 0, x2 - x1, h),
        Paint()..color = bgColor);
    }

    // Fill gradient
    final fillPath = Path();
    for (int i = 0; i < data.length; i++) {
      final x = (i / (data.length - 1)) * size.width;
      final y = h - ((values[i] - minV) / range) * h;
      if (i == 0) fillPath.moveTo(x, y);
      else fillPath.lineTo(x, y);
    }
    fillPath.lineTo(size.width, h);
    fillPath.lineTo(0, h);
    fillPath.close();
    canvas.drawPath(fillPath, Paint()
      ..shader = LinearGradient(colors: [
        Colors.purple.withOpacity(0.4),
        Colors.purple.withOpacity(0.05)],
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
      ).createShader(Rect.fromLTWH(0, 0, size.width, h)));

    // Line
    final linePath = Path();
    for (int i = 0; i < data.length; i++) {
      final x = (i / (data.length - 1)) * size.width;
      final y = h - ((values[i] - minV) / range) * h;
      if (i == 0) linePath.moveTo(x, y);
      else linePath.lineTo(x, y);
    }
    canvas.drawPath(linePath, Paint()
      ..color = Colors.purpleAccent
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke);

    // Dots with labels
    for (int i = 0; i < data.length; i += 6) {
      final x = (i / (data.length - 1)) * size.width;
      final y = h - ((values[i] - minV) / range) * h;
      canvas.drawCircle(Offset(x, y), 4, Paint()
        ..color = Colors.purpleAccent
        ..style = PaintingStyle.fill);
      final tp = TextPainter(
        text: TextSpan(
          text: '${data[i]['hour']}:00\n'
                '${values[i].toStringAsFixed(1)}kW',
          style: const TextStyle(
            color: Colors.white70, fontSize: 9)),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(
        (x - tp.width / 2).clamp(0, size.width - tp.width),
        y - 30));
    }
  }

  @override
  bool shouldRepaint(_LSTMChartPainter old) => old.data != data;
}