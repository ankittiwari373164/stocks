// ============================================================
//  DROP-IN for your existing server.js
//  Replaces the Yahoo-scraping MTF prediction with the real
//  Groww-API predictor running as a Python service on :8000.
//  Paste this near your other app.get(...) routes.
// ============================================================
const PREDICTOR_URL = process.env.PREDICTOR_URL || 'http://localhost:8000';

// Most-traded prediction (cached JSON from the Python predictor)
app.get('/api/mtf/predictions', async (req, res) => {
  try {
    const r = await axios.get(`${PREDICTOR_URL}/predict`, { timeout: 8000 });
    res.json(r.data);
  } catch (e) {
    res.status(503).json({ error: 'predictor offline', detail: e.message });
  }
});

// Trigger a fresh live prediction (call this at/after 09:25 IST)
app.post('/api/mtf/analyze', async (req, res) => {
  try {
    const r = await axios.post(`${PREDICTOR_URL}/predict/run`, {}, { timeout: 120000 });
    res.json(r.data);
  } catch (e) {
    res.status(503).json({ error: 'predictor offline', detail: e.message });
  }
});

// Backtest accuracy summary
app.get('/api/mtf/backtest', async (req, res) => {
  try {
    const r = await axios.get(`${PREDICTOR_URL}/backtest`, { timeout: 120000 });
    res.json(r.data);
  } catch (e) {
    res.status(503).json({ error: 'predictor offline', detail: e.message });
  }
});
