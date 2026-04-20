function app() {
  return {
    // Navigation
    tabs: [
      { id: 'upload', label: 'Upload' },
      { id: 'chunking', label: 'Chunking' },
      { id: 'coding', label: 'Coding' },
      { id: 'analysis', label: 'Analysis' },
    ],
    currentTab: 'upload',

    // Uploads
    uploads: [],

    // Chunking
    chunkOpts: { method: 'llm', similarity_threshold: 0.5, min_sentences: 1, max_sentences: 8, remove_timestamps: true, preserve_speakers: true },
    chunks: [],
    chunkRunning: false,
    chunkLog: [],

    // Codebook
    codebookSource: 'custom',
    customCodesText: '',
    codebook: {},
    codebookBuilding: false,
    codebookSaved: false,
    codebookLog: [],

    // Coding
    approach: 'hybrid',
    codedRows: [],
    codingDone: false,
    codeRunning: false,
    codeLog: [],

    // Analysis
    themes: '',
    themesRunning: false,
    themeLog: [],

    // ----------------------------------------------------------------
    // Init
    // ----------------------------------------------------------------
    async init() {
      await this.loadUploads()
      await this.loadChunks()
      await this.loadCodebook()
      await this.loadCodingResults()
      await this.loadThemes()
    },

    // ----------------------------------------------------------------
    // Uploads
    // ----------------------------------------------------------------
    async loadUploads() {
      const res = await fetch('/api/upload')
      if (res.ok) {
        const data = await res.json()
        const seen = new Set()
        this.uploads = data.filter(u => {
          const key = u.role + '::' + u.filename
          if (seen.has(key)) return false
          seen.add(key)
          return true
        })
      }
    },

    handleFileInput(event, role) {
      for (const file of event.target.files) {
        this.uploadFile(file, role)
      }
    },

    handleDrop(event, role) {
      for (const file of event.dataTransfer.files) {
        this.uploadFile(file, role)
      }
    },

    async deleteUpload(role, filename) {
      const res = await fetch(`/api/upload/${role}/${encodeURIComponent(filename)}`, { method: 'DELETE' })
      if (res.ok) {
        this.uploads = this.uploads.filter(u => !(u.role === role && u.filename === filename))
      }
    },

    async uploadFile(file, role) {
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch(`/api/upload/${role}`, { method: 'POST', body: fd })
      if (res.ok) {
        const data = await res.json()
        this.uploads = this.uploads.filter(u => !(u.role === role && u.filename === data.filename))
        this.uploads.push(data)
      } else {
        alert(`Upload failed: ${await res.text()}`)
      }
    },

    // ----------------------------------------------------------------
    // Chunking
    // ----------------------------------------------------------------
    async runChunking() {
      this.chunkRunning = true
      this.chunkLog = ['Starting chunking…']
      this.chunks = []
      try {
        const es = await this._ssePost('/api/chunk', this.chunkOpts)
        es.onmessage = (e) => {
          const payload = JSON.parse(e.data)
          if (payload.type === 'progress') {
            this.chunkLog.push(payload.message)
          } else if (payload.type === 'done') {
            this.chunks = payload.chunks || []
            this.chunkLog.push(`✓ Done — ${this.chunks.length} chunks`)
            es.close()
            this.chunkRunning = false
          }
        }
        es.onerror = () => {
          this.chunkLog.push('Connection closed')
          this.chunkRunning = false
          es.close()
        }
      } catch (e) {
        this.chunkLog.push(`Error: ${e}`)
        this.chunkRunning = false
      }
    },

    async loadChunks() {
      const res = await fetch('/api/chunk/results')
      if (res.ok) this.chunks = await res.json()
    },

    // ----------------------------------------------------------------
    // Codebook
    // ----------------------------------------------------------------
    async saveCustomCodes() {
      const res = await fetch('/api/codebook/custom', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: this.customCodesText }),
      })
      if (res.ok) {
        const data = await res.json()
        await this.loadCodebook()
        this.codebookSaved = true
        setTimeout(() => this.codebookSaved = false, 2000)
      }
    },

    async buildCodebookFromGuide() {
      this.codebookBuilding = true
      this.codebookLog = ['Extracting codes from interview guide…']
      const es = await this._ssePost('/api/codebook/from-guide', {})
      es.onmessage = async (e) => {
        const payload = JSON.parse(e.data)
        if (payload.type === 'progress') this.codebookLog.push(payload.message)
        if (payload.type === 'done') {
          this.codebookLog.push(`✓ Extracted ${Object.keys(payload.codebook || {}).length} codes`)
          await this.loadCodebook()
          es.close()
          this.codebookBuilding = false
        }
      }
      es.onerror = () => { this.codebookBuilding = false; es.close() }
    },

    async buildCodebookFromLiterature() {
      this.codebookBuilding = true
      this.codebookLog = ['Building codebook from literature…']
      const es = await this._ssePost('/api/codebook/build', {})
      es.onmessage = async (e) => {
        const payload = JSON.parse(e.data)
        if (payload.type === 'progress') this.codebookLog.push(payload.message)
        if (payload.type === 'done') {
          this.codebookLog.push(`✓ Built ${Object.keys(payload.codebook || {}).length} codes`)
          await this.loadCodebook()
          es.close()
          this.codebookBuilding = false
        }
      }
      es.onerror = () => { this.codebookBuilding = false; es.close() }
    },

    async importCodebook(event) {
      const file = event.target.files[0]
      if (!file) return
      const fd = new FormData()
      fd.append('file', file)
      const res = await fetch('/api/codebook/import', { method: 'POST', body: fd })
      if (res.ok) {
        await this.loadCodebook()
        this.codebookSaved = true
        setTimeout(() => this.codebookSaved = false, 2000)
      }
    },

    async loadCodebook() {
      const res = await fetch('/api/codebook')
      if (res.ok) this.codebook = await res.json()
    },

    // ----------------------------------------------------------------
    // Coding
    // ----------------------------------------------------------------
    async runCoding() {
      this.codeRunning = true
      this.codeLog = ['Starting coding…']
      this.codedRows = []
      const es = await this._ssePost('/api/code', { approach: this.approach })
      es.onmessage = (e) => {
        const payload = JSON.parse(e.data)
        if (payload.type === 'progress') this.codeLog.push(payload.message)
        if (payload.type === 'deductive_done') this.codeLog.push(`✓ Deductive coding complete`)
        if (payload.type === 'inductive_done') this.codeLog.push(`✓ Inductive coding complete`)
        if (payload.type === 'done') {
          this.codedRows = payload.preview || []
          this.codingDone = true
          this.codeLog.push(`✓ Coded ${payload.total_chunks} chunks`)
          es.close()
          this.codeRunning = false
        }
      }
      es.onerror = async () => {
        this.codeRunning = false
        es.close()
        await this.loadCodingResults()
      }
    },

    async loadCodingResults() {
      const res = await fetch('/api/code/results')
      if (res.ok) {
        this.codedRows = await res.json()
        if (this.codedRows.length > 0) this.codingDone = true
      }
    },

    // ----------------------------------------------------------------
    // Themes
    // ----------------------------------------------------------------
    async runThemes() {
      this.themesRunning = true
      this.themeLog = ['Generating thematic analysis…']
      this.themes = ''
      const es = await this._ssePost('/api/code/themes', {})
      es.onmessage = (e) => {
        const payload = JSON.parse(e.data)
        if (payload.type === 'progress') this.themeLog.push(payload.message)
        if (payload.type === 'done') {
          this.themes = payload.themes || ''
          this.themeLog.push('✓ Done')
          es.close()
          this.themesRunning = false
        }
      }
      es.onerror = () => { this.themesRunning = false; es.close() }
    },

    async loadThemes() {
      const res = await fetch('/api/code/themes')
      if (res.ok) {
        const data = await res.json()
        this.themes = data.themes || ''
      }
    },

    // ----------------------------------------------------------------
    // SSE helper — POST then open EventSource via GET with session cookie
    // ----------------------------------------------------------------
    async _ssePost(url, body) {
      // Trigger the POST which returns a StreamingResponse
      // We use fetch with ReadableStream to handle SSE over POST
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!response.ok) {
        const text = await response.text()
        throw new Error(`${response.status}: ${text}`)
      }

      // Wrap response stream in an EventSource-like object
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      const listeners = { message: null, error: null }
      const fake = {
        set onmessage(fn) { listeners.message = fn },
        set onerror(fn) { listeners.error = fn },
        close() { reader.cancel() },
      }

      async function pump() {
        try {
          while (true) {
            const { done, value } = await reader.read()
            buffer += done
              ? decoder.decode()              // flush encoder state
              : decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = done ? '' : lines.pop()  // on done, process everything
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                listeners.message?.({ data: line.slice(6) })
              }
            }
            if (done) break
          }
        } catch (e) {
          listeners.error?.(e)
        }
      }

      pump()
      return fake
    },
  }
}
