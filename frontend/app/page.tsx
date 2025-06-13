'use client'

import { useState } from 'react'
import axios from 'axios'

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<string>('')
  const [result, setResult] = useState<any>(null)

  const handleUpload = async () => {
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    setStatus('Uploading...')
    try {
      const res = await axios.post('/api/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      if (res.data.status === 'completed') {
        setResult(res.data.data)
        setStatus('Completed')
      } else {
        setStatus('Error: ' + res.data.error)
      }
    } catch (e: any) {
      setStatus('Error: ' + e.message)
    }
  }

  return (
    <div style={{ maxWidth: 600, margin: '2rem auto', fontFamily: 'sans-serif' }}>
      <h1>Upload Field Document</h1>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />
      <button onClick={handleUpload} disabled={!file} style={{ marginLeft: '1rem' }}>
        Upload
      </button>
      <p>{status}</p>
      {result && (
        <pre style={{ background: '#eee', padding: '1rem' }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  )
}
