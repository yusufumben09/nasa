const express = require('express');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { weatherFor, waypointImagesFor, trekLayers } = require('./weather');

const app = express();
const port = process.env.PORT || 3000;
const aiBackendUrl = process.env.AI_BACKEND_URL || 'http://localhost:5001';
const aiTimeoutMs = Number(process.env.AI_TIMEOUT_MS) > 0 ? Number(process.env.AI_TIMEOUT_MS) : 300000;
const pathingModulePromise = import(pathToFileURL(path.join(__dirname, 'pathing/src/server/routing.js')).href);
const pathingTerrainPromise = pathingModulePromise.then(({ Terrain }) => Terrain.load(path.join(__dirname, 'pathing/src/build')));
const systemPrompt = 'You are a helpful assistant. Respond clearly and accurately.';
const navigationSystemPrompt = [
    'You are a practical Mars rover route analyst. Write all user-facing text in English.',
    'The route has already been calculated by weighted Dijkstra search. Do not modify the path or invent coordinates.',
    'If the route is marked as a least-risk fallback, state plainly that no fully passable route exists and that this alternative still crosses hazards. Explain the measured fallback risk; never call it safe or passable.',
    'Return one overview, not one explanation per waypoint. Explain why this route is preferred using only the supplied verified metrics: traversal cost, distance, surface distance, mean and maximum slope, ascent, descent, elevation, passability, depression/crater avoidance cost, nearby alternative tradeoff, and data source coverage.',
    'Explain that weighted Dijkstra minimizes the fixed balanced rover cost on this sampled grid, with a one-cell safety buffer around extreme or unknown terrain; it does not guarantee the shortest physical distance, best weather, or real-world safety. Compare the route factors clearly and mention endpoint adjustments when provided.',
    'If risk.shortestAlternative is present, compare its measured distance, balanced traversal cost, depression penalty, and adjacent blocked-cell contacts with the chosen route. Describe only differences supported by those values; do not assume the alternative is riskier. If it is absent, say the shortest and balanced objectives produced the same route.',
    'environment.weather holds measured medians for the selected local hour at Jezero (Perseverance MEDA temperature, pressure, wind and humidity; Mastcam-Z dust optical depth; Mars24 sun elevation; Curiosity RAD radiation from Gale crater). Describe them as typical measured conditions, not a forecast. Use only these supplied values; if a value is null, say it is unavailable. Soil strength is unavailable. Never invent weather conditions. Weather did not change the route geometry.',
    'Include a concise navigation summary describing the broad sequence of route directions and practical rover-following tips, without repeating every coordinate.',
    'Return exactly one syntactically valid JSON object with this schema: {"why":"specific English explanation","factors":[{"name":"Slope","value":"verified value or comparison","effect":"why it matters"}],"environment":{"slope":"...","elevation":"...","surface":"...","weather":"short summary using only the supplied weather values"},"navigation":"concise English overview and practical tips"}.',
    'Every object property must have a colon between its name and value. Separate properties and array items with commas. Quote all property names and string values with double quotes. Do not use comments, trailing commas, or unquoted text.',
    'The required structure is illustrated by this valid example (numbers are illustrative only; use the supplied values): {"why":"This route balances slope and climb under the fixed rover cost. It does not guarantee the shortest distance; typical measured conditions are listed below.","factors":[{"name":"Slope","value":"mean 8.2°, max 18.4°","effect":"Lower slope reduces climbing effort and slip risk."}],"environment":{"slope":"mean 8.2°","elevation":"ascent 42 m","surface":"horizontal 1200 m, surface 1210 m","weather":"air -40.0 °C, wind median 4.1 m/s"},"navigation":"Follow the route in order; confirm heading on long straight legs and slow down at turns."}.',
    'Use only numbers that appear in the supplied JSON (you may convert metres to kilometres). Use a dot as the decimal separator.',
    'Output only the JSON object. Do not include markdown, code fences, greetings, analysis, or text outside it.'
].join(' ');

app.use(express.json({ limit: '10mb' }));

async function requestAi(messages, instructions = systemPrompt, jsonMode = false, maxTokens) {
    const aiResponse = await fetch(`${aiBackendUrl.replace(/\/+$/, '')}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            ...(process.env.AI_MODEL ? { model: process.env.AI_MODEL } : {}),
            ...(jsonMode ? { response_format: { type: 'json_object' } } : {}),
                messages: [{ role: 'system', content: instructions }, ...messages],
                ...(maxTokens ? { max_tokens: maxTokens } : {}),
        }),
        signal: AbortSignal.timeout(aiTimeoutMs)
    });
    const responseBody = await aiResponse.text();
    let responseData;

    try {
        responseData = JSON.parse(responseBody);
    } catch {
        throw Object.assign(new Error('AI backend returned an invalid JSON response'), { status: 502 });
    }

    if (!aiResponse.ok) {
        throw Object.assign(new Error('AI backend request failed'), { status: 502, details: responseData });
    }

    return responseData;
}

function sendAiError(res, error) {
    if (error.name === 'TimeoutError') {
        return res.status(504).json({ error: 'AI backend request timed out' });
    }
    if (error.message === 'fetch failed') {
        return res.status(502).json({
            error: 'Unable to connect to llama.cpp. Verify that its OpenAI-compatible server is running and AI_BACKEND_URL points to it.'
        });
    }
    return res.status(error.status || 502).json({ error: error.message || 'Unable to reach AI backend', details: error.details });
}

function parseAssistantJson(responseData) {
    const content = responseData.choices?.[0]?.message?.content;
    if (typeof content !== 'string') throw new Error('AI response did not contain text');
    const jsonText = content.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
    return JSON.parse(jsonText);
}

// --- Narration number check (soft version, PR #1 review) ---
// Only numbers WITH a physical unit (°, °C, K, m, km, m/s, Pa, %, mSv, mGy) are checked; unitless numbers
// (counts, hours like 13:00, "8-connected", Ls, sol numbers) are never rejected.
// A unit number passes if it is within 2% (or 0.05 absolute) of a fact with a matching unit/key, also after m <-> km
// conversion, so rounding like "about 36 km" is fine. Facts = JSON sent to the model WITHOUT route coordinates.
// If a sentence contains an unsupported unit number, ONLY that sentence is removed; the rest of the model text stays.
// The whole narration is replaced by the template only if its structure is invalid or "why" becomes empty.
const UNIT_NUMBER = /(-?\d[\d.,]*\d|-?\d)\s*(°\s?C|°|K\b|km\b|m\/s|m\b|Pa\b|%|mSv|mGy)/g;
// Which fact keys a unit may be compared with (unit-aware, so "27.3°" is only checked against angles).
const UNIT_KEYS = [
    [/^°\s?C$/, /_c$|temp/],
    [/^K$/, /_k$|temp/],
    [/^°$/, /deg|slope|angle/],
    [/^km$|^m$/, /_m$|_m_|distance|ascent|descent|elevation_(start|end)/],
    [/^m\/s$/, /wind/],
    [/^Pa$/, /pa$|pressure/],
    [/^%$/, /pct|share|percent|humidity/],
    [/^mSv$/, /msv/],
    [/^mGy$/, /mgy/]
];
function factNumbers(facts) {
    const out = [];
    const walk = (v, key) => {
        if (typeof v === 'number' && Number.isFinite(v)) out.push({ key, v });
        else if (Array.isArray(v)) v.forEach(x => walk(x, key));
        else if (v && typeof v === 'object') Object.entries(v).forEach(([k, x]) => walk(x, key ? `${key}.${k}` : k));
    };
    walk(facts, '');
    return out.map(x => ({ key: x.key.toLowerCase(), v: x.v }));
}
function candidatesFor(unit, facts) {
    const u = unit.replace(/\s/g, '');
    const rule = UNIT_KEYS.find(([re]) => re.test(u));
    const pick = rule ? facts.filter(f => rule[1].test(f.key)) : facts;
    const vals = pick.map(f => f.v);
    if (u === 'km') return vals.flatMap(v => [v, v / 1000]);
    if (u === 'm') return vals.flatMap(v => [v, v * 1000]);
    return vals;
}
function tokenReadings(token) {
    const readings = new Set();
    const en = Number(token.replace(/,/g, ''));
    if (Number.isFinite(en)) readings.add(en);
    const tr = Number(token.replace(/\./g, '').replace(',', '.'));
    if (Number.isFinite(tr)) readings.add(tr);
    return [...readings];
}
function numberSupported(token, unit, facts) {
    const known = candidatesFor(unit, facts);
    return tokenReadings(token).some(v => known.some(k => {
        const a = Math.abs(v), b = Math.abs(k);
        return Math.abs(a - b) <= Math.max(0.051, 0.02 * b);
    }));
}
function unsupportedIn(text, known) {
    const bad = [];
    for (const m of String(text).matchAll(UNIT_NUMBER)) if (!numberSupported(m[1], m[2], known)) bad.push(m[0].trim());
    return bad;
}
function filterText(text, known, problems) {
    if (typeof text !== 'string') return text;
    const sentences = text.split(/(?<=[.!?])\s+/);
    const kept = sentences.filter(sentence => {
        const bad = unsupportedIn(sentence, known);
        if (bad.length) problems.push(`removed sentence with unsupported value(s): ${bad.join(', ')}`);
        return !bad.length;
    });
    return kept.join(' ').trim();
}
function weatherSummary(w) {
    if (!w || !w.available) return 'not available';
    const parts = [];
    if (Number.isFinite(w.air_temp_c)) parts.push(`air ${w.air_temp_c} °C`);
    if (Number.isFinite(w.ground_temp_c)) parts.push(`ground ${w.ground_temp_c} °C`);
    if (Number.isFinite(w.pressure_pa)) parts.push(`pressure ${w.pressure_pa} Pa`);
    if (w.wind_m_s) parts.push(`wind median ${w.wind_m_s.median} m/s`);
    if (w.dust_optical_depth) parts.push(`dust optical depth ${w.dust_optical_depth.median}`);
    return `Typical measured values at ${w.hour_ltst}:00 local time: ${parts.join(', ')}.`;
}
function templateNarration(facts) {
    const e = facts.environment || {};
    const o = facts.optimality || {};
    const fmt = (v, d = 1) => (Number.isFinite(v) ? Number(v.toFixed(d)) : 'n/a');
    const fallbackNote = facts.fallback
        ? 'No fully passable route exists for these points; this is the least-risk alternative and it still crosses hazardous terrain. It is not safe.'
        : 'All cells on this route are within the slope limit used by the planner.';
    return {
        why: `${fallbackNote} The route was computed by ${o.algorithm || 'the route engine'}; the language model did not change it.`,
        factors: [
            { name: 'Distance', value: `${fmt(e.distance_m / 1000, 2)} km (surface ${fmt(e.surface_distance_m / 1000, 2)} km)`, effect: 'Longer routes take more time and energy.' },
            { name: 'Slope', value: `mean ${fmt(e.mean_slope_deg)}°, max ${fmt(e.max_slope_deg)}°`, effect: 'Steeper ground raises slip and tip-over risk.' },
            { name: 'Elevation change', value: `ascent ${fmt(e.ascent_m, 0)} m, descent ${fmt(e.descent_m, 0)} m`, effect: 'Climbing costs energy; descending steep ground needs care.' }
        ],
        environment: { slope: `mean ${fmt(e.mean_slope_deg)}°`, elevation: `ascent ${fmt(e.ascent_m, 0)} m`, surface: `horizontal ${fmt(e.distance_m, 0)} m, surface ${fmt(e.surface_distance_m, 0)} m`, weather: weatherSummary(e.weather) },
        navigation: 'Follow the route in order, confirm heading on long straight legs and slow down at turns.'
    };
}
function checkedNarration(narration, facts) {
    if (!narration || typeof narration.why !== 'string' || !Array.isArray(narration.factors) ||
        !narration.environment || typeof narration.environment !== 'object' || typeof narration.navigation !== 'string') {
        return { narration: templateNarration(facts), source: 'template', problems: ['invalid narration structure'] };
    }
    const known = factNumbers(facts);
    const problems = [];
    const out = {
        why: filterText(narration.why, known, problems),
        factors: narration.factors
            .map(f => (f && typeof f === 'object' ? { ...f, value: filterText(f.value, known, problems), effect: filterText(f.effect, known, problems) } : f))
            .filter(f => f && typeof f.name === 'string' && f.value),
        environment: Object.fromEntries(Object.entries(narration.environment).map(([k, v]) => [k, filterText(v, known, problems)])),
        navigation: filterText(narration.navigation, known, problems)
    };
    if (!out.why) return { narration: templateNarration(facts), source: 'template', problems: [...problems, 'why empty after filtering'] };
    return { narration: out, source: problems.length ? 'model_filtered' : 'model', problems };
}

function fallbackNavigationStep(waypoint) {
    const directions = { east: 'doğuya', west: 'batıya', south: 'güneye', north: 'kuzeye' };
    const direction = directions[waypoint.legDirection];
    const legMoves = waypoint.legGridMoves;
    return {
        instruction: direction
            ? `${direction} ilerleyerek (${waypoint.x}, ${waypoint.y}) waypointine ulaş.`
            : `Başlangıç waypointi (${waypoint.x}, ${waypoint.y}) konumunu doğrula.`,
        commentary: direction
            ? `Bu bacak ${direction} yönünde ${legMoves} ızgara hareketi sürer ve hesaplanan düşük maliyetli rover rotasını izler.`
            : 'Bu, hesaplanan düşük maliyetli rover rotasının başlangıç noktasıdır.',
        tip: legMoves > 10
            ? 'Uzun düz bacakta yönünü düzenli doğrula; sonraki dönüş waypointine yaklaşırken yavaşla.'
            : 'Waypoint koordinatını doğrula ve yön değişiminde kontrollü ilerle.',
        x: waypoint.x,
        y: waypoint.y
    };
}

function normalizeNavigationSteps(steps, waypoints, optimality) {
    return waypoints.map((waypoint, index) => {
        const step = steps[index];
        const fallback = fallbackNavigationStep(waypoint);
        return {
            instruction: typeof step?.instruction === 'string' && step.instruction.trim()
                ? step.instruction : fallback.instruction,
            commentary: typeof step?.commentary === 'string' && step.commentary.trim()
                ? step.commentary : fallback.commentary,
            tip: typeof step?.tip === 'string' && step.tip.trim()
                ? step.tip : fallback.tip,
            x: waypoint.x,
            y: waypoint.y
        };
    });
}

function frontendPointToLonLat(point, terrain) {
    const bounds = terrain.meta.grid.bounds_lonlat;
    return {
        lon: bounds.west + (point.x + 0.5) / 176 * (bounds.east - bounds.west),
        lat: bounds.north - (point.y + 0.5) / 176 * (bounds.north - bounds.south)
    };
}

function lonLatToFrontendPoint(coordinate, terrain) {
    const bounds = terrain.meta.grid.bounds_lonlat;
    return {
        x: Math.max(0, Math.min(175, Math.round((coordinate[0] - bounds.west) / (bounds.east - bounds.west) * 176 - 0.5))),
        y: Math.max(0, Math.min(175, Math.round((bounds.north - coordinate[1]) / (bounds.north - bounds.south) * 176 - 0.5)))
    };
}

function makePathingTerrain(terrain, Terrain) {
    const size = terrain.width * terrain.height;
    const slope = new Float64Array(size);
    const elevation = new Float64Array(size);
    for (let y = 0; y < terrain.height; y++) {
        for (let x = 0; x < terrain.width; x++) {
            const source = terrain.cells[y][x];
            const index = y * terrain.width + x;
            elevation[index] = source[0];
            slope[index] = source[1];
        }
    }
    const meta = {
        grid: {
            sphere_radius_m: 180 / Math.PI,
            lat_ts_deg: 0,
            cell_map_m: [1, 1],
            origin_map_m: { x_left: 0, y_top: terrain.height },
            bounds_lonlat: null,
            crs_proj4: 'synthetic sampled map grid'
        }
    };
    return new Terrain({
        H: terrain.height,
        W: terrain.width,
        slopeMax: slope,
        slopeMean: slope,
        elev: elevation,
        dx: new Float64Array(terrain.height).fill(1),
        dy: 1
    }, meta);
}

function mapPointToPathing(point, terrain) {
    const x = point.x * (terrain.width - 1) / 175;
    const y = point.y * (terrain.height - 1) / 175;
    return { lon: x + 0.5, lat: terrain.height - y - 0.5 };
}

function pathingCoordinateToMap(coordinate, terrain) {
    return {
        x: Math.round((coordinate[0] - 0.5) * 175 / (terrain.width - 1)),
        y: Math.round((terrain.height - coordinate[1] - 0.5) * 175 / (terrain.height - 1))
    };
}

function routeWaypoints(route) {
    const points = route.filter((point, index) => index === 0 || point.x !== route[index - 1].x || point.y !== route[index - 1].y);
    const indices = [0];
    let previousDirection = null;
    for (let index = 1; index < points.length; index++) {
        const direction = { x: points[index].x - points[index - 1].x, y: points[index].y - points[index - 1].y };
        if (previousDirection && (direction.x !== previousDirection.x || direction.y !== previousDirection.y)) indices.push(index - 1);
        previousDirection = direction;
    }
    if (points.length > 1) indices.push(points.length - 1);
    return [...new Set(indices)].map((index, waypointIndex, uniqueIndices) => {
        const point = points[index];
        const previous = waypointIndex ? points[uniqueIndices[waypointIndex - 1]] : null;
        const dx = previous ? point.x - previous.x : 0;
        const dy = previous ? point.y - previous.y : 0;
        return {
            ...point,
            legDirection: !previous ? 'start' : Math.abs(dx) >= Math.abs(dy) ? (dx >= 0 ? 'east' : 'west') : (dy >= 0 ? 'south' : 'north'),
            legGridMoves: previous ? Math.abs(dx) + Math.abs(dy) : 0
        };
    });
}

function routeProfile(coordinates, terrain) {
    let distance = 0;
    const distance_m = [];
    const elevation_m = [];
    const slope_deg = [];
    const cells = [];
    for (let index = 0; index < coordinates.length; index++) {
        const [lon, lat] = coordinates[index];
        const [row, col] = terrain.cellOf(lon, lat);
        const cell = row * terrain.W + col;
        if (index) {
            const [previousLon, previousLat] = coordinates[index - 1];
            distance += terrain.groundDistance(previousLon, previousLat, lon, lat);
        }
        distance_m.push(Math.round(distance * 10) / 10);
        elevation_m.push(terrain.elev[cell]);
        slope_deg.push(terrain.slopeMax[cell]);
        cells.push({ row, col, x: Math.round((col + 0.5) * 176 / terrain.W - 0.5), y: Math.round((row + 0.5) * 176 / terrain.H - 0.5) });
    }
    return { distance_m, elevation_m, slope_deg, cells };
}

function routeRisk(route, terrain) {
    const cells = new Set(route.geometry.coordinates.map(([lon, lat]) => {
        const [row, col] = terrain.cellOf(lon, lat);
        return row * terrain.W + col;
    }));
    let blockedAdjacentContacts = 0;
    let depressionPenaltyTotal = 0;
    for (const cell of cells) {
        const row = Math.floor(cell / terrain.W), col = cell % terrain.W;
        let adjacentToBlocked = false;
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (!dr && !dc) continue;
                const r = row + dr, c = col + dc;
                if (r >= 0 && r < terrain.H && c >= 0 && c < terrain.W && !terrain.passable[r * terrain.W + c]) {
                    adjacentToBlocked = true;
                }
            }
        }
        if (adjacentToBlocked) blockedAdjacentContacts++;
        depressionPenaltyTotal += terrain.depressionPenalty[cell];
    }
    return {
        blockedAdjacentContacts,
        meanDepressionPenalty: cells.size ? Math.round(depressionPenaltyTotal / cells.size * 1000) / 1000 : 0,
        routeCells: cells.size
    };
}

const ROUTE_RULES = Object.freeze({ slopeLimit: 20, rockLimit: 0.85 });

function passableCellsNear(terrain, target, radius = 3) {
    const cells = [];
    const minX = Math.max(0, Math.floor(target.x - radius));
    const maxX = Math.min(terrain.width - 1, Math.ceil(target.x + radius));
    const minY = Math.max(0, Math.floor(target.y - radius));
    const maxY = Math.min(terrain.height - 1, Math.ceil(target.y + radius));
    for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
            if ((x - target.x) ** 2 + (y - target.y) ** 2 > radius ** 2) continue;
            const cell = terrain.cells[y][x];
            if (cell[1] > ROUTE_RULES.slopeLimit || cell[2] > ROUTE_RULES.rockLimit) continue;
            cells.push({ x, y, distance: (x - target.x) ** 2 + (y - target.y) ** 2 });
        }
    }
    return cells.sort((a, b) => a.distance - b.distance);
}

function findGridRoute(terrain, start, end) {
    const toGrid = point => ({
        x: Math.round(point.x * (terrain.width - 1) / 175),
        y: Math.round(point.y * (terrain.height - 1) / 175)
    });
    const startCandidates = passableCellsNear(terrain, toGrid(start));
    const endCandidates = passableCellsNear(terrain, toGrid(end));
    if (!startCandidates.length || !endCandidates.length) return null;

    const total = terrain.width * terrain.height;
    const stateCount = total * 4;
    const distances = new Float64Array(stateCount).fill(Infinity);
    const parents = new Int32Array(stateCount).fill(-1);
    const endIndices = new Set(endCandidates.map(point => point.y * terrain.width + point.x));
    const heap = [];
    const directions = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    const push = item => {
        heap.push(item);
        let index = heap.length - 1;
        while (index > 0) {
            const parent = Math.floor((index - 1) / 2);
            if (heap[parent].cost <= item.cost) break;
            heap[index] = heap[parent];
            index = parent;
        }
        heap[index] = item;
    };
    const pop = () => {
        const result = heap[0];
        const last = heap.pop();
        if (heap.length) {
            let index = 0;
            while (index * 2 + 1 < heap.length) {
                let child = index * 2 + 1;
                if (child + 1 < heap.length && heap[child + 1].cost < heap[child].cost) child++;
                if (heap[child].cost >= last.cost) break;
                heap[index] = heap[child];
                index = child;
            }
            heap[index] = last;
        }
        return result;
    };
    for (const point of startCandidates) {
        const cell = point.y * terrain.width + point.x;
        for (let direction = 0; direction < 4; direction++) {
            const state = cell * 4 + direction;
            distances[state] = 0;
            parents[state] = state;
            push({ state, cost: 0 });
        }
    }

    let endIndex = -1;
    while (heap.length) {
        const current = pop();
        if (current.cost !== distances[current.state]) continue;
        const currentCell = Math.floor(current.state / 4);
        const previousDirection = current.state % 4;
        if (endIndices.has(currentCell)) {
            endIndex = current.state;
            break;
        }
        const x = currentCell % terrain.width;
        const y = Math.floor(currentCell / terrain.width);
        for (const [dx, dy] of directions) {
            const nextX = x + dx;
            const nextY = y + dy;
            if (nextX < 0 || nextY < 0 || nextX >= terrain.width || nextY >= terrain.height) continue;
            const direction = directions.findIndex(([xStep, yStep]) => xStep === dx && yStep === dy);
            const next = nextY * terrain.width + nextX;
            const cell = terrain.cells[nextY][nextX];
            if (cell[1] > ROUTE_RULES.slopeLimit || cell[2] > ROUTE_RULES.rockLimit) continue;
            const previousCell = terrain.cells[y][x];
            const uphill = Math.max(0, cell[0] - previousCell[0]);
            const slopeCost = 4 * (cell[1] / ROUTE_RULES.slopeLimit) ** 2;
            const surfaceCost = 5 * cell[2] + 8 * cell[3];
            const climbCost = uphill / 10;
            const turnCost = previousDirection === direction ? 0 : 1.5;
            const nextState = next * 4 + direction;
            const nextCost = current.cost + 1 + slopeCost + surfaceCost + climbCost + turnCost;
            if (nextCost < distances[nextState]) {
                distances[nextState] = nextCost;
                parents[nextState] = current.state;
                push({ state: nextState, cost: nextCost });
            }
        }
    }
    if (endIndex === -1) return null;

    const gridPath = [];
    for (let current = endIndex; ; current = parents[current]) {
        const cell = Math.floor(current / 4);
        gridPath.push({ x: cell % terrain.width, y: Math.floor(cell / terrain.width) });
        if (parents[current] === current) break;
    }
    gridPath.reverse();
    const route = gridPath.map(point => ({
        x: Math.round(point.x * 175 / (terrain.width - 1)),
        y: Math.round(point.y * 175 / (terrain.height - 1))
    }));
    const waypointIndices = [0];
    let previousDirection = null;
    for (let index = 1; index < gridPath.length; index++) {
        const direction = {
            x: gridPath[index].x - gridPath[index - 1].x,
            y: gridPath[index].y - gridPath[index - 1].y
        };
        if (previousDirection && (direction.x !== previousDirection.x || direction.y !== previousDirection.y)) {
            waypointIndices.push(index - 1);
        }
        previousDirection = direction;
    }
    if (gridPath.length > 1) waypointIndices.push(gridPath.length - 1);
    const uniqueIndices = [...new Set(waypointIndices)];
    const waypoints = uniqueIndices.map((index, waypointIndex) => {
        const point = gridPath[index];
        const previous = waypointIndex > 0 ? gridPath[uniqueIndices[waypointIndex - 1]] : null;
        const dx = previous ? point.x - previous.x : 0;
        const dy = previous ? point.y - previous.y : 0;
        const legDirection = !previous ? 'start' : Math.abs(dx) >= Math.abs(dy)
            ? (dx >= 0 ? 'east' : 'west')
            : (dy >= 0 ? 'south' : 'north');
        return {
            ...route[index],
            legDirection,
            legGridMoves: previous ? Math.abs(dx) + Math.abs(dy) : 0
        };
    });
    const selectedStart = route[0];
    const selectedEnd = route[route.length - 1];
    return {
        route,
        waypoints,
        optimality: {
            algorithm: 'weighted Dijkstra search',
            criterion: 'lowest fixed rover traversal cost using slope, rock, sand, uphill travel, and turn penalties',
            gridMoves: gridPath.length - 1,
            traversalCost: Number(distances[endIndex].toFixed(2)),
            gridWidth: terrain.width,
            gridHeight: terrain.height,
            slopeLimit: ROUTE_RULES.slopeLimit,
            rockLimit: ROUTE_RULES.rockLimit,
            startAdjusted: selectedStart.x !== start.x || selectedStart.y !== start.y,
            endAdjusted: selectedEnd.x !== end.x || selectedEnd.y !== end.y
        }
    };
}

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

app.get('/terrain-model', (req, res) => {
    res.json({
        file: '/jezero.blend',
        status: 'unverified',
        routing_used: false,
        reason: 'The Blender asset has no verified lon/lat and elevation transform. Routing uses the verified CTX/HiRISE terrain build instead.'
    });
});

app.get('/jezero.blend', (req, res) => {
    res.sendFile(path.join(__dirname, 'jezero.blend'));
});

app.get('/image.png', (req, res) => {
    res.sendFile(path.join(__dirname, 'image.png'));
});

app.get('/terrain', async (req, res) => {
    try {
        const terrain = await pathingTerrainPromise;
        const width = 176;
        const height = 176;
        const renderWidth = 528;
        const renderHeight = 528;
        const sample = (array, x, y, outWidth, outHeight) => {
            const sourceX = Math.max(0, Math.min(terrain.W - 1, (x + 0.5) * terrain.W / outWidth - 0.5));
            const sourceY = Math.max(0, Math.min(terrain.H - 1, (y + 0.5) * terrain.H / outHeight - 0.5));
            const x0 = Math.floor(sourceX), y0 = Math.floor(sourceY);
            const x1 = Math.min(terrain.W - 1, x0 + 1), y1 = Math.min(terrain.H - 1, y0 + 1);
            const tx = sourceX - x0, ty = sourceY - y0;
            const a = array[y0 * terrain.W + x0], b = array[y0 * terrain.W + x1];
            const c = array[y1 * terrain.W + x0], d = array[y1 * terrain.W + x1];
            return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
        };
        const gridSample = (outWidth, outHeight, array) => Array.from({ length: outHeight }, (_, y) =>
            Array.from({ length: outWidth }, (_, x) => sample(array, x, y, outWidth, outHeight)));
        const elevation = gridSample(width, height, terrain.elev);
        const slope = gridSample(width, height, terrain.slopeMax);
        return res.json({
            width,
            height,
            bounds: terrain.meta.grid.bounds_lonlat,
            elevation,
            slope,
            renderWidth,
            renderHeight,
            renderElevation: gridSample(renderWidth, renderHeight, terrain.elev),
            renderSlope: gridSample(renderWidth, renderHeight, terrain.slopeMax),
            source: 'pathing/src/build',
            routing_surface: 'CTX 20 m + HiRISE-derived limiting slope',
            extreme_slope_rule: 'Cells with limiting slope above 20 degrees are impassable.'
        });
    } catch (error) {
        return res.status(500).json({ error: 'Unable to load real terrain data', details: error.message });
    }
});

app.post('/ai', async (req, res) => {
    const { prompt, image } = req.body;
    if (typeof prompt !== 'string' || !prompt.trim()) {
        return res.status(400).json({ error: 'Prompt is required' });
    }
    if (image !== undefined && typeof image !== 'string') {
        return res.status(400).json({ error: 'Image must be a URL or image data URL' });
    }
    if (image && !/^https?:\/\/|^data:image\//i.test(image)) {
        return res.status(400).json({ error: 'Image must be an HTTP(S) URL or image data URL' });
    }

    const content = image
        ? [
            { type: 'text', text: prompt.trim() },
            { type: 'image_url', image_url: { url: image } }
        ]
        : prompt.trim();

    try {
        return res.json(await requestAi([{ role: 'user', content }]));
    } catch (error) {
        return sendAiError(res, error);
    }
});

app.post('/route', async (req, res) => {
    const { start, end } = req.body;
    const hour = Number.isFinite(Number(req.body.hour)) ? Number(req.body.hour) : 13;
    const ls = Number.isFinite(Number(req.body.ls)) && req.body.ls !== null && req.body.ls !== '' ? Number(req.body.ls) : null;
    const validPoint = point => point && Number.isFinite(point.x) && Number.isFinite(point.y) &&
        point.x >= 0 && point.x < 176 && point.y >= 0 && point.y < 176;
    if (!validPoint(start) || !validPoint(end)) {
        return res.status(400).json({ error: 'Start and end must be inside the map bounds' });
    }

    try {
        const { SLOPE_LIMIT_DEG } = await pathingModulePromise;
        const pathingTerrain = await pathingTerrainPromise;
        const pathingResult = pathingTerrain.compute(
            frontendPointToLonLat(start, pathingTerrain),
            frontendPointToLonLat(end, pathingTerrain)
        );
        if (pathingResult.status !== 'ok' && pathingResult.status !== 'fallback') {
            return res.status(422).json({
                error: pathingResult.reason?.message || 'No traversable route exists for the rover safety limit',
                status: pathingResult.status,
                details: pathingResult.errors || pathingResult.reason || null
            });
        }
        const balanced = pathingResult.routes.find(route => route.objective === 'balanced') || pathingResult.routes[0];
        const route = balanced.geometry.coordinates.map(coordinate => lonLatToFrontendPoint(coordinate, pathingTerrain));
        const waypoints = routeWaypoints(route);
        const environment = {
            distance_m: balanced.distance_m,
            surface_distance_m: balanced.surface_distance_m,
            mean_slope_deg: balanced.mean_slope_deg,
            max_slope_deg: balanced.max_slope_deg,
            ascent_m: balanced.ascent_m,
            descent_m: balanced.descent_m,
            elevation_start_m: balanced.geometry.vertex_elevation_m[0],
            elevation_end_m: balanced.geometry.vertex_elevation_m.at(-1),
            cells: balanced.cells,
            hirise_cell_share_pct: balanced.hirise_cell_share_pct,
            weather: weatherFor({ hour, ls }),
            soil_strength: 'unavailable; not used in routing'
        };
        const profile = routeProfile(balanced.geometry.coordinates, pathingTerrain);
        const shortest = pathingResult.routes.find(route => route.objective === 'shortest');
        const risk = {
            chosen: routeRisk(balanced, pathingTerrain),
            shortestAlternative: shortest && shortest !== balanced ? {
                distance_m: shortest.distance_m,
                mean_slope_deg: shortest.mean_slope_deg,
                traversalCost: shortest.cost_breakdown.balanced_cost,
                ...routeRisk(shortest, pathingTerrain)
            } : null
        };
        if (risk.shortestAlternative) {
            risk.shortestAlternative.distanceDifferenceFromChosen_m = Math.round((shortest.distance_m - balanced.distance_m) * 10) / 10;
            risk.shortestAlternative.traversalCostDifferenceFromChosen = Math.round((shortest.cost_breakdown.balanced_cost - balanced.cost_breakdown.balanced_cost) * 10000) / 10000;
        }
        const optimality = {
            algorithm: pathingResult.status === 'fallback' ? 'pathing Terrain least-risk fallback Dijkstra' : 'pathing Terrain balanced Dijkstra',
            criterion: pathingResult.status === 'fallback'
                ? 'lowest hazardous exposure available when no fully passable route exists; terrain violations are reported'
                : 'lowest fixed rover traversal cost using 8-connected moves, slope, DEM depression avoidance, one-cell obstacle buffer, and no-corner-cutting rules',
            gridMoves: Math.max(0, balanced.cells - 1),
            traversalCost: balanced.cost_breakdown.balanced_cost,
            slopeLimit: SLOPE_LIMIT_DEG,
            startAdjusted: route[0].x !== start.x || route[0].y !== start.y,
            endAdjusted: route[route.length - 1].x !== end.x || route[route.length - 1].y !== end.y,
            engine: 'pathing/src/server/routing.js',
            depression_avoidance: 'multi-scale DEM depression penalty discourages crater-like basins; extreme slopes remain impassable',
            safety_buffer: 'one cell around extreme-slope or unknown terrain is also impassable',
            data: 'pathing/src/build (CTX + HiRISE-derived terrain arrays)'
        };
        const response = {
            route,
            waypoints,
            environment,
            profile,
            risk,
            optimality,
            status: pathingResult.status,
            fallback: pathingResult.status === 'fallback',
            routeWarning: pathingResult.status === 'fallback' ? pathingResult.reason?.message : null,
            validation: balanced.validation
        };
        try {
            const aiResponse = await requestAi([{
                role: 'user',
                content: JSON.stringify({
                    task: 'Explain why this computed rover route is preferred using a single evidence-based overview.',
                    route, optimality, environment, risk,
                    status: pathingResult.status,
                    fallback: pathingResult.status === 'fallback',
                    routeWarning: response.routeWarning,
                    validation: balanced.validation
                })
            }], navigationSystemPrompt, true, 1800);
            const facts = { optimality, environment, risk, status: pathingResult.status, fallback: response.fallback, validation: balanced.validation };
            let parsed = null;
            try { parsed = parseAssistantJson(aiResponse); } catch { parsed = null; }
            const checked = checkedNarration(parsed, facts);
            response.narration = checked.narration;
            response.narrationSource = checked.source;
            response.narrationCheck = checked.problems;
        } catch (error) {
            response.narrationError = error.message || 'AI narration unavailable';
        }
        return res.json(response);
    } catch (error) {
        console.error('Pathing engine failed:', error);
        return res.status(500).json({ error: 'Pathing engine failed', details: error.message });
    }
});

app.post('/ai/narrate', async (req, res) => {
    const { route, optimality, environment, risk } = req.body;
    if (!Array.isArray(route) || route.length < 2 || route.length > 2000 ||
        route.some(point => !point || !Number.isFinite(point.x) || !Number.isFinite(point.y) ||
            point.x < 0 || point.x > 175 || point.y < 0 || point.y > 175) ||
        !optimality || typeof optimality.criterion !== 'string' ||
        !environment || typeof environment !== 'object') {
        return res.status(400).json({ error: 'Invalid route overview data' });
    }

    try {
        const aiResponse = await requestAi([{
            role: 'user',
            content: JSON.stringify({
                task: 'Explain why this computed rover route is preferred using a single evidence-based overview.',
                optimality,
                environment,
                risk: risk || null,
                route
            })
        }], navigationSystemPrompt, true, 1800);
        let parsed = null;
        try { parsed = parseAssistantJson(aiResponse); } catch { parsed = null; }
        const checked = checkedNarration(parsed, { optimality, environment, risk: risk || null });
        return res.json({ ...checked.narration, source: checked.source, check: checked.problems });
    } catch (error) {
        return sendAiError(res, error);
    }
});

// --- NASA data routes (data/ folder) ---
app.get('/weather', (req, res) => {
    res.json(weatherFor({ hour: req.query.hour ?? 13, ls: req.query.ls ?? null }));
});
app.get('/waypoint-images', (req, res) => {
    res.json(waypointImagesFor(req.query.site, req.query.drive));
});
app.get('/trek-layers', (req, res) => {
    res.json(trekLayers || { status: 'not_ready', layers: [] });
});




app.listen(port, () => {
    console.log(`Server listening on port ${port}`);
});

