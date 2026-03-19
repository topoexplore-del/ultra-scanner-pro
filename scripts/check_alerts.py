"""
ULTRA SCANNER PRO — Multi-Layer Alert System v5.0
Validates ALL 4 layers before sending alerts via Email + Telegram.
Includes holding period recommendations and ETF/Index-specific evaluation.

Run: python scripts/check_alerts.py
"""
import json, os, smtplib, warnings, sys
import urllib.request, urllib.parse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")

# ═══ HELPERS ═══════════════════════════════════════════════════
def grade_score(gr):
    if not gr: return 0
    g = gr.lower()
    if g in ('excel','strong','cheap'): return 5
    if g in ('good','solid'): return 4
    if g in ('fair','mod'): return 3
    if g in ('med','pricey'): return 2
    return 1 if g != 'n/a' else 0

def composite_score(r):
    raw = (grade_score(r.get('eps_gr',''))*0.30 + grade_score(r.get('roe_gr',''))*0.25 +
           grade_score(r.get('roa_gr',''))*0.20 + grade_score(r.get('pe_gr',''))*0.25)
    return round((raw/5)*100)

def is_etf_or_index(r):
    s = (r.get('sector') or '').lower()
    return s in ('index','etf','commodity','fixed income','')

def bayesian_prob(r, cs):
    ai = r.get('ai') or 50; rsi = r.get('rsi') or 50; adx = r.get('adx') or 15
    d20 = r.get('20d') or 0; rv = r.get('rel_vol') or 1
    p = 0.5 * (ai/100) * (1.3 if rv>1.1 else 0.9) * (1.2 if 40<rsi<70 else 0.7)
    p *= (cs/85) * (1.2 if d20>3 else 1.0 if d20>0 else 0.7) * (1.15 if adx>20 else 0.85) * 3.5
    return min(0.95, max(0.15, p))

# ═══ HOLDING PERIOD CALCULATOR ═════════════════════════════════
def calc_holding_period(r, cs):
    """Determine optimal holding period based on signal strength and momentum."""
    score = r.get('score', 0); ai = r.get('ai', 0); d20 = r.get('20d', 0) or 0
    d5 = r.get('5d', 0) or 0; upside = r.get('upside', 0) or 0
    state = r.get('state', 'WAIT'); etf = is_etf_or_index(r)

    # Strong short-term signals → shorter hold
    # Weaker signals but good fundamentals → longer hold
    periods = []

    if state in ('ENTRY+','ENTRY') and d5 > 2 and ai > 70:
        periods.append({'period': '1 Semana', 'target_pct': '3-6%', 'confidence': 'Alta',
                       'reason': 'Señal activa con momentum corto fuerte'})

    if cs >= 60 and d20 > 5 and upside >= 8:
        periods.append({'period': '1 Mes', 'target_pct': '6-10%', 'confidence': 'Alta',
                       'reason': 'Fundamentales sólidos + tendencia 20D alcista'})

    if cs >= 55 and upside >= 10:
        periods.append({'period': '3 Meses', 'target_pct': '8-15%', 'confidence': 'Media-Alta',
                       'reason': 'Composite fuerte con upside significativo'})

    if cs >= 50 and upside >= 12 and not etf:
        periods.append({'period': '6 Meses', 'target_pct': '12-20%', 'confidence': 'Media',
                       'reason': 'Valor fundamental con potencial de revaluación'})

    if cs >= 65 and upside >= 15 and not etf:
        periods.append({'period': '1 Año', 'target_pct': '15-30%', 'confidence': 'Media',
                       'reason': 'Tesis de largo plazo con fundamentales excelentes'})

    # ETFs get different periods
    if etf and d20 > 3:
        periods.append({'period': '1-3 Meses', 'target_pct': '5-12%', 'confidence': 'Media',
                       'reason': 'ETF/Índice con tendencia favorable'})

    if not periods:
        periods.append({'period': '1 Mes', 'target_pct': '3-6%', 'confidence': 'Baja',
                       'reason': 'Señal mínima — monitorear de cerca'})

    # Best period = first (most confident)
    return periods

# ═══ 4-LAYER VALIDATION ════════════════════════════════════════
def validate_all(r):
    etf = is_etf_or_index(r)
    cs = composite_score(r)
    result = {'ticker': r.get('ticker','?'), 'name': r.get('name','?'), 'price': r.get('close',0),
              'sector': r.get('sector',''), 'state': r.get('state','WAIT'), 'score': r.get('score',0),
              'ai': r.get('ai',0), 'composite': cs, 'upside': r.get('upside',0) or 0,
              'target': r.get('target',0), 'roe': r.get('roe',0) or 0, 'roa': r.get('roa',0) or 0,
              'eps_g': r.get('eps_g',0) or 0, 'pe': r.get('pe',0), 'd20': r.get('20d',0) or 0,
              'd5': r.get('5d',0) or 0, 'rsi': r.get('rsi',50), 'is_etf': etf,
              'all_passed': False, 'layers_passed': 0, 'group': '', 'failed': []}

    # Layer 1: RADAR
    l1 = (r.get('state','') in ('ENTRY+','ENTRY') and r.get('score',0) >= 60 and
          (r.get('ai',0)) >= 65 and r.get('abc','') in ('A','B') and
          (r.get('rsi') is None or r.get('rsi',50) < 75))
    if not l1: result['failed'].append('Radar')

    # Layer 2: ANALYSIS (ETFs bypass fundamental checks)
    if etf:
        l2 = r.get('score',0) >= 30 and (r.get('20d',0) or 0) > -5
    else:
        eq_clean = not (grade_score(r.get('eps_gr',''))>=4 and grade_score(r.get('roa_gr',''))<=2)
        debt_ok = not ((r.get('roe',0) or 0)>0 and (r.get('roa',0) or 0)>0 and (r.get('roe',0))/(r.get('roa',0))>3)
        l2 = cs >= 55 and (r.get('eps_g',0) or 0) > 0 and (r.get('roe',0) or 0) >= 8 and (r.get('roa',0) or 0) >= 3 and eq_clean and debt_ok
    if not l2: result['failed'].append('Analysis')

    # Layer 3: ENTRY ZONES
    upside = r.get('upside',0) or 0
    l3 = upside >= 8 and (r.get('20d',0) or 0) > 0 and (r.get('5d',0) or 0) > -5
    if etf: l3 = upside >= 5 and (r.get('20d',0) or 0) > -2  # Softer for ETFs
    if not l3: result['failed'].append('Entry Zones')

    # Layer 4: GAME THEORY
    prob = bayesian_prob(r, cs); prob_pct = round(prob*100)
    ev = (prob * 3) - ((1-prob) * 2)
    kelly = max(0, prob - (1-prob) / 1.5)
    l4 = prob_pct >= 65 and ev > 0 and kelly >= 0.10
    if etf: l4 = prob_pct >= 55 and ev > 0  # Softer for ETFs
    if not l4: result['failed'].append('Game Theory')

    result['layers_passed'] = sum([l1,l2,l3,l4])
    result['all_passed'] = l1 and l2 and l3 and l4
    result['prob'] = prob_pct
    result['ev'] = round(ev, 2)
    result['kelly'] = round(kelly*100)

    # Entry zones
    close = r.get('close',0) or 1
    result['zones'] = {'entry': round(close*0.985,2), 'tp1': round(close*1.03,2),
                       'tp2': round(close*1.06,2), 'sl': round(close*0.98,2)}

    # Holding periods
    result['periods'] = calc_holding_period(r, cs)

    return result

# ═══ TELEGRAM BOT API ══════════════════════════════════════════
def send_telegram(message, token, chat_id):
    """Send Telegram message via Bot API."""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=15)
        if resp.status == 200:
            print(f"  ✅ Telegram sent to chat {chat_id}")
        else:
            print(f"  ⚠️ Telegram status: {resp.status}")
    except Exception as e:
        print(f"  ❌ Telegram error: {e}")

def build_telegram_msg(confirmed, near_miss, timestamp):
    """Build Telegram message with Markdown."""
    msg = f"🎯 *ULTRA SCANNER PRO*\n{timestamp}\n"
    msg += f"━━━━━━━━━━━━━━━━━\n"
    msg += f"✅ *{len(confirmed)} SEÑAL(ES) 4/4*\n\n"

    for a in confirmed:
        z = a['zones']; p = a['periods'][0] if a['periods'] else {}
        msg += f"🚀 *{a['ticker']}* — COMPRAR\n"
        msg += f"💰 Precio: ${a['price']}\n"
        msg += f"📊 CS:{a['composite']} | AI:{a['ai']}% | P(↑):{a['prob']}%\n"
        msg += f"🎯 Entry: ${z['entry']} → TP1: ${z['tp1']} | TP2: ${z['tp2']}\n"
        msg += f"🛑 SL: ${z['sl']} | Upside: +{a['upside']}%\n"
        msg += f"⏱ *Mantener: {p.get('period','1 Mes')}* ({p.get('target_pct','3-6%')})\n"
        msg += f"📋 {p.get('reason','')}\n\n"

    if near_miss:
        msg += f"⏳ *WATCHLIST (3/4):*\n"
        for a in near_miss[:5]:
            msg += f"  • {a['ticker']} CS:{a['composite']} — falta: {', '.join(a['failed'])}\n"

    msg += f"\n⚠️ No es consejo financiero"
    return msg

# ═══ EMAIL ═════════════════════════════════════════════════════
def build_email_html(confirmed, near_miss, timestamp):
    html = f"""<html><body style="margin:0;padding:0;background:#0a0a0f;font-family:'Segoe UI',Arial,sans-serif;color:#c8cad0">
    <div style="max-width:640px;margin:0 auto;padding:20px">
    <div style="background:#12131a;border:1px solid #2a2b36;border-radius:8px;padding:20px;margin-bottom:16px;text-align:center">
        <h1 style="color:#00e676;margin:0;font-size:22px;letter-spacing:2px">🎯 ULTRA SCANNER PRO</h1>
        <p style="color:#8a8c96;margin:6px 0 0;font-size:12px">ALERTA MULTI-CAPA · {timestamp}</p>
        <div style="margin:12px 0 0;padding:10px;background:#0a2e1a;border:1px solid #00e67640;border-radius:6px">
            <p style="color:#00e676;margin:0;font-size:16px;font-weight:bold">{len(confirmed)} SEÑAL(ES) CONFIRMADA(S)</p>
            <p style="color:#69f0ae;margin:4px 0 0;font-size:10px">4/4 CAPAS: Radar ✓ Analysis ✓ Entry Zones ✓ Game Theory ✓</p>
        </div>
    </div>"""

    for a in confirmed:
        z = a['zones']; etf_tag = " (ETF/Índice)" if a['is_etf'] else ""
        html += f"""
    <div style="background:#12131a;border:1px solid #00e676;border-left:4px solid #00e676;border-radius:8px;padding:16px;margin-bottom:12px">
        <div style="margin-bottom:12px">
            <div style="font-size:18px;font-weight:bold;color:#00e676">🚀 {a['ticker']} — {a['name']}{etf_tag}</div>
            <div style="font-size:11px;color:#ffab00;font-weight:bold;margin-top:4px">COMPRAR — 4/4 CAPAS CONFIRMADAS · {a['sector']}</div>
        </div>
        <table style="width:100%;border-collapse:separate;border-spacing:4px"><tr>
            <td style="padding:8px;background:#1a1b24;border-radius:4px;text-align:center"><div style="font-size:9px;color:#5a5c66">PRECIO</div><div style="font-size:15px;font-weight:bold;color:#c8cad0">${a['price']}</div></td>
            <td style="padding:8px;background:#1a1b24;border-radius:4px;text-align:center"><div style="font-size:9px;color:#5a5c66">CS</div><div style="font-size:15px;font-weight:bold;color:#00e676">{a['composite']}</div></td>
            <td style="padding:8px;background:#1a1b24;border-radius:4px;text-align:center"><div style="font-size:9px;color:#5a5c66">AI</div><div style="font-size:15px;font-weight:bold;color:#00e5ff">{a['ai']}%</div></td>
            <td style="padding:8px;background:#1a1b24;border-radius:4px;text-align:center"><div style="font-size:9px;color:#5a5c66">PROB</div><div style="font-size:15px;font-weight:bold;color:#b388ff">{a['prob']}%</div></td>
            <td style="padding:8px;background:#1a1b24;border-radius:4px;text-align:center"><div style="font-size:9px;color:#5a5c66">UPSIDE</div><div style="font-size:15px;font-weight:bold;color:#00e676">+{a['upside']}%</div></td>
        </tr></table>
        <div style="background:#0a2e1a;border:1px solid #00e67640;border-radius:6px;padding:12px;margin:10px 0">
            <div style="font-size:10px;color:#00e676;margin-bottom:8px">🎯 ZONAS DE OPERACIÓN</div>
            <table style="width:100%;border-collapse:separate;border-spacing:4px"><tr>
                <td style="padding:8px;background:#12131a;border-radius:4px;text-align:center"><div style="font-size:9px;color:#5a5c66">ENTRY</div><div style="font-size:14px;font-weight:bold;color:#00e676">${z['entry']}</div></td>
                <td style="padding:8px;background:#12131a;border-radius:4px;text-align:center"><div style="font-size:9px;color:#5a5c66">TP1 (+3%)</div><div style="font-size:14px;font-weight:bold;color:#00e5ff">${z['tp1']}</div></td>
                <td style="padding:8px;background:#12131a;border-radius:4px;text-align:center"><div style="font-size:9px;color:#5a5c66">TP2 (+6%)</div><div style="font-size:14px;font-weight:bold;color:#69f0ae">${z['tp2']}</div></td>
                <td style="padding:8px;background:#12131a;border-radius:4px;text-align:center"><div style="font-size:9px;color:#5a5c66">STOP LOSS</div><div style="font-size:14px;font-weight:bold;color:#ff5252">${z['sl']}</div></td>
            </tr></table>
        </div>
        <div style="background:#1a1024;border:1px solid #b388ff40;border-radius:6px;padding:12px;margin:10px 0">
            <div style="font-size:10px;color:#b388ff;margin-bottom:6px">⏱ PERIODOS RECOMENDADOS</div>"""

        for p in a['periods']:
            conf_color = '#00e676' if p['confidence']=='Alta' else '#ffab00' if 'Media' in p['confidence'] else '#ff5252'
            html += f"""<div style="padding:4px 0;border-bottom:1px solid #2a2b36;font-size:11px">
                <span style="color:{conf_color};font-weight:bold">{p['period']}</span>
                <span style="color:#c8cad0"> → {p['target_pct']}</span>
                <span style="color:#5a5c66"> · Conf: {p['confidence']}</span>
                <div style="color:#8a8c96;font-size:10px;margin-top:2px">{p['reason']}</div>
            </div>"""

        html += f"""</div>
        <div style="background:#1a1b24;border-radius:4px;padding:8px;font-size:10px;color:#8a8c96">
            ✓ Radar: {a['state']} Score:{a['score']} · ✓ CS:{a['composite']} ROE:{a['roe']}% · ✓ EV:+{a['ev']}% Kelly:{a['kelly']}%
        </div>
    </div>"""

    if near_miss:
        html += '<div style="background:#12131a;border:1px solid #2a2b36;border-radius:8px;padding:14px;margin-bottom:12px">'
        html += '<p style="color:#ffab00;font-size:12px;font-weight:bold;margin:0 0 8px">⏳ WATCHLIST — 3/4 Capas</p>'
        for a in near_miss[:8]:
            html += f'<div style="padding:4px 0;border-bottom:1px solid #2a2b36;font-size:11px"><span style="color:#ffab00;font-weight:bold">{a["ticker"]}</span> <span style="color:#8a8c96">${a["price"]} CS:{a["composite"]}</span> <span style="color:#ff5252;font-size:10px">falta: {", ".join(a["failed"])}</span></div>'
        html += '</div>'

    html += f"""<div style="text-align:center;padding:16px;color:#5a5c66;font-size:10px;border-top:1px solid #2a2b36">
        <p style="margin:0">ULTRA SCANNER PRO v5.0 · Solo LONG · 4-Layer Validation</p>
        <p style="margin:4px 0 0;color:#ff5252;font-weight:bold">⚠️ No es consejo financiero.</p>
    </div></div></body></html>"""
    return html

def send_email(confirmed, near_miss, to_email, smtp_user, smtp_pass, timestamp):
    msg = MIMEMultipart('alternative')
    tickers = [a['ticker'] for a in confirmed]
    msg['Subject'] = f"🚀 COMPRAR: {', '.join(tickers)} — {len(confirmed)} señal(es) 4/4"
    msg['From'] = smtp_user; msg['To'] = to_email
    msg.attach(MIMEText(build_telegram_msg(confirmed, near_miss, timestamp), 'plain', 'utf-8'))
    msg.attach(MIMEText(build_email_html(confirmed, near_miss, timestamp), 'html', 'utf-8'))
    server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls()
    server.login(smtp_user, smtp_pass); server.sendmail(smtp_user, to_email, msg.as_string()); server.quit()
    print(f"  ✅ Email sent to {to_email}")

# ═══ MAIN ══════════════════════════════════════════════════════
def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'snapshot.json')
    if not os.path.exists(data_path):
        print("ERROR: snapshot.json not found."); sys.exit(1)
    with open(data_path, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)

    bogota = timezone(timedelta(hours=-5))
    timestamp = datetime.now(bogota).strftime("%Y-%m-%d %H:%M") + " (Bogotá)"
    print(f"ULTRA SCANNER PRO — Alert Check · {timestamp}\n{'='*55}")

    confirmed, near_miss = [], []
    for gn, stocks in snapshot.get('groups', {}).items():
        for r in stocks:
            res = validate_all(r); res['group'] = gn
            if res['all_passed']:
                confirmed.append(res)
                p = res['periods'][0] if res['periods'] else {}
                print(f"  🚀 {res['ticker']} — 4/4 ✓ CS:{res['composite']} Prob:{res['prob']}% → {p.get('period','?')} ({p.get('target_pct','?')})")
            elif res['layers_passed'] == 3:
                near_miss.append(res)

    confirmed.sort(key=lambda x: (-x['composite'], -x['prob']))
    near_miss.sort(key=lambda x: -x['composite'])
    print(f"\n{'='*55}\nRESULTS: {len(confirmed)} confirmed | {len(near_miss)} watchlist (3/4)")

    # Save
    alerts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'alerts.json')
    with open(alerts_path, 'w') as f:
        json.dump({'checked_at': timestamp, 'confirmed': len(confirmed),
                   'signals': [{'ticker':a['ticker'],'composite':a['composite'],'prob':a['prob'],
                               'zones':a['zones'],'periods':[p['period'] for p in a['periods']]}
                              for a in confirmed],
                   'watchlist': [{'ticker':a['ticker'],'failed':a['failed']} for a in near_miss[:15]]
                  }, f, indent=2)

    smtp_user = os.environ.get('SMTP_USER',''); smtp_pass = os.environ.get('SMTP_PASS','')
    to_email = os.environ.get('ALERT_EMAIL','andrestf88@gmail.com')
    tg_token = os.environ.get('TELEGRAM_TOKEN','')
    tg_chat = os.environ.get('TELEGRAM_CHAT_ID','')

    if confirmed:
        # Email
        if smtp_user and smtp_pass:
            try: send_email(confirmed, near_miss, to_email, smtp_user, smtp_pass, timestamp)
            except Exception as e: print(f"  ❌ Email error: {e}")
        else:
            print(f"\n  ⚠️ SMTP not configured. Set SMTP_USER + SMTP_PASS secrets.")

        # Telegram
        if tg_token and tg_chat:
            try:
                tg_msg = build_telegram_msg(confirmed, near_miss, timestamp)
                send_telegram(tg_msg, tg_token, tg_chat)
            except Exception as e: print(f"  ❌ Telegram error: {e}")
        else:
            print(f"  ⚠️ Telegram not configured. Set TELEGRAM_TOKEN + TELEGRAM_CHAT_ID secrets.")

        # Print signals
        print(f"\n  Señales confirmadas:")
        for a in confirmed:
            z = a['zones']; p = a['periods']
            print(f"    🚀 {a['ticker']} ${a['price']} → E:${z['entry']} TP1:${z['tp1']} TP2:${z['tp2']} SL:${z['sl']}")
            for pp in p:
                print(f"       ⏱ {pp['period']} ({pp['target_pct']}) — {pp['reason']}")
    else:
        print(f"\n  ℹ️ No signals met ALL 4 layers.")
        if near_miss:
            print(f"  Watchlist (3/4):")
            for a in near_miss[:5]:
                print(f"    ⏳ {a['ticker']} CS:{a['composite']} — falta: {', '.join(a['failed'])}")

if __name__ == "__main__":
    main()
