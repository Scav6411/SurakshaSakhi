import { useState, useRef, useEffect } from 'react'
import { Mic, MicOff, X, Stethoscope, Languages, Loader } from 'lucide-react'
import { Patient, PSURow } from '../types'
import { genieQuery, GenieResponse } from '../api/client'

interface Props {
  patients: Patient[]
  psuData:  PSURow[]
}

type Lang = 'en-IN' | 'hi-IN'

const LABELS: Record<Lang, { btn: string; listening: string; placeholder: string; toggle: string; thinking: string }> = {
  'en-IN': { btn: '🎤', listening: 'Listening…', placeholder: 'Ask about patients, villages, or risk…', toggle: 'हिंदी', thinking: 'Thinking…' },
  'hi-IN': { btn: '🎤', listening: 'सुन रहा हूँ…', placeholder: 'मरीजों, गांव या जोखिम के बारे में पूछें…', toggle: 'English', thinking: 'सोच रहा हूँ…' },
}

export default function VoiceAssistant({ patients, psuData }: Props) {
  const [open,           setOpen]           = useState(false)
  const [lang,           setLang]           = useState<Lang>('hi-IN')
  const [listening,      setListening]      = useState(false)
  const [loading,        setLoading]        = useState(false)
  const [transcript,     setTranscript]     = useState('')
  const [response,       setResponse]       = useState('')
  const [tableData,      setTableData]      = useState<{ columns: string[]; rows: string[][] } | null>(null)
  const [supported,      setSupported]      = useState(true)

  const recogRef    = useRef<any>(null)
  const convIdRef   = useRef<string | undefined>(undefined)

  useEffect(() => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) { setSupported(false); return }
    const r = new SR()
    r.continuous     = false
    r.interimResults = false
    r.lang           = lang
    r.onresult = async (e: any) => {
      const text = e.results[0][0].transcript
      setTranscript(text)
      setTableData(null)
      setResponse('')
      setLoading(true)
      try {
        const res: GenieResponse = await genieQuery(text, convIdRef.current)
        convIdRef.current = res.conversation_id
        setResponse(res.text)
        if (res.columns && res.rows) {
          setTableData({ columns: res.columns, rows: res.rows })
        }
        speak(res.text, lang)
      } catch {
        const fallback = lang === 'hi-IN'
          ? 'माफ करें, अभी जानकारी नहीं मिल पाई।'
          : 'Sorry, could not get an answer right now.'
        setResponse(fallback)
        speak(fallback, lang)
      } finally {
        setLoading(false)
      }
    }
    r.onend  = () => setListening(false)
    r.onerror = () => setListening(false)
    recogRef.current = r
  }, [lang])

  const speak = (text: string, l: Lang) => {
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(text)
    u.lang = l
    u.rate = 0.9
    window.speechSynthesis.speak(u)
  }

  const startListening = () => {
    if (!recogRef.current || loading) return
    setTranscript('')
    setResponse('')
    setTableData(null)
    recogRef.current.lang = lang
    recogRef.current.start()
    setListening(true)
  }

  const lbl = LABELS[lang]

  if (!supported) return null

  return (
    <div className="va-wrapper">
      {open && (
        <div className="va-panel">
          <div className="va-panel-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Stethoscope size={15} /> Suraksha Assistant
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className="va-lang-btn"
                onClick={() => { setLang(l => l === 'hi-IN' ? 'en-IN' : 'hi-IN'); convIdRef.current = undefined }}
                style={{ display: 'flex', alignItems: 'center', gap: 4 }}
              >
                <Languages size={13} /> {lbl.toggle}
              </button>
              <button className="va-close" onClick={() => setOpen(false)}><X size={16} /></button>
            </div>
          </div>

          <div className="va-body">
            {!transcript && !response && !loading && (
              <p className="va-hint">{lbl.placeholder}</p>
            )}

            {transcript && (
              <div className="va-bubble va-user">
                <span>🎙 {transcript}</span>
              </div>
            )}

            {loading && (
              <div className="va-bubble va-bot va-loading">
                <Loader size={14} className="va-spin" />
                <span>{lbl.thinking}</span>
              </div>
            )}

            {!loading && response && (
              <div className="va-bubble va-bot">
                <span>🤖 {response}</span>
              </div>
            )}

            {!loading && tableData && (
              <div className="va-table-wrap">
                <table className="va-table">
                  <thead>
                    <tr>
                      {tableData.columns.map(col => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tableData.rows.map((row, i) => (
                      <tr key={i}>
                        {row.map((cell, j) => <td key={j}>{cell ?? '—'}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="va-panel-footer">
            <p className="va-suggestions">
              {lang === 'hi-IN'
                ? '"उच्च जोखिम मरीज दिखाओ" · "PSU 5 की महिलाएं" · "सबसे खतरनाक गांव?"'
                : '"Show top high risk patients" · "Women in PSU 5" · "Most critical village?"'}
            </p>
            <button
              className={`va-mic-btn ${listening ? 'va-listening' : ''}`}
              onClick={startListening}
              disabled={listening || loading}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
            >
              {listening
                ? <><MicOff size={16} /> {lbl.listening}</>
                : loading
                  ? <><Loader size={16} className="va-spin" /> {lbl.thinking}</>
                  : <><Mic size={16} /> Speak</>}
            </button>
          </div>
        </div>
      )}

      <button className={`va-fab ${listening ? 'va-fab-listening' : ''}`} onClick={() => setOpen(o => !o)}>
        {listening ? <MicOff size={22} /> : <Mic size={22} />}
      </button>
    </div>
  )
}
