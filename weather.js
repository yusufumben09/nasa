'use strict';
// weather.js — data/ klasorundeki NASA verilerini (MEDA, Mastcam-Z, Mars24, Raw Images) sunucuya verir.
// Dosya yoksa ya da bozuksa sunucu cokmez; ilgili alan "unavailable" doner.
const fs = require('node:fs');
const path = require('node:path');

const DATA = path.join(__dirname, 'data');
function load(name) {
    try { return JSON.parse(fs.readFileSync(path.join(DATA, name), 'utf8')); }
    catch (error) { console.warn(`data/${name} could not be read: ${error.message}`); return null; }
}
const env = load('environment_meda.json');
const sun = load('sun_elevation.json');
const wpImages = load('waypoint_images.json');
const trek = load('trek_layers.json');

const EVA_START = 11;
const EVA_END = 15; // 11:00-15:00 LTST

function lsBin(ls) {
    return String(Math.floor((((ls % 360) + 360) % 360) / 30) * 30);
}
function nearestSeason(ls) {
    const s = [0, 90, 180, 270, 360].reduce((a, b) => (Math.abs(b - ls) < Math.abs(a - ls) ? b : a));
    return String(s % 360);
}

/**
 * Secilen saat (LTST, 0-23) ve istege bagli mevsim (Ls, derece) icin cevre ozeti.
 * Butun sayilar data/ dosyalarindan gelir; hesap yapilan tek yer derece->Celsius ve yuvarlama.
 */
function weatherFor({ hour = 13, ls = null } = {}) {
    if (!env) return { available: false, note: 'environment data file missing' };
    const h = Math.max(0, Math.min(23, Math.round(Number(hour))));
    const hk = String(h);
    const t = (env.temperature?.by_ltst_hour || []).find(x => x.hour === h);
    const ps = env.pressure?.by_hour_ltst?.[hk];
    const ws = env.wind?.by_hour_ltst?.[hk];
    const rh = env.humidity?.by_hour_ltst?.[hk];
    const hasLs = Number.isFinite(Number(ls)) && ls !== null && ls !== '';
    const lsNum = hasLs ? Number(ls) : null;
    const dust = hasLs ? env.dust_opacity?.by_ls_30deg?.[lsBin(lsNum)] : env.dust_opacity?.all;
    let sunElev = null;
    if (sun?.by_ls) {
        const key = `${String(h).padStart(2, '0')}:00`;
        if (hasLs) sunElev = { season_ls: Number(nearestSeason(lsNum)), elevation_deg: sun.by_ls[nearestSeason(lsNum)]?.[key] ?? null };
        else {
            const vals = Object.values(sun.by_ls).map(v => v[key]).filter(Number.isFinite);
            sunElev = vals.length ? { min_deg: Math.min(...vals), max_deg: Math.max(...vals), note: 'range over seasons Ls 0/90/180/270' } : null;
        }
    }
    const rad = env.radiation || {};
    return {
        available: true,
        source: 'Perseverance MEDA (temperature, pressure, wind, humidity), Mastcam-Z (dust), Mars24 (sun), Curiosity RAD (radiation, Gale crater)',
        hour_ltst: h,
        ls_deg: lsNum,
        in_eva_window: h >= EVA_START && h < EVA_END,
        eva_window_start_hour: EVA_START,
        eva_window_end_hour: EVA_END,
        ground_temp_c: t ? t.ground_temp_k.median_c : null,
        ground_temp_k: t ? t.ground_temp_k.median : null,
        air_temp_c: t ? t.air_temp_k.median_c : null,
        air_temp_k: t ? t.air_temp_k.median : null,
        pressure_pa: ps ? ps.median : null,
        wind_m_s: ws ? { median: ws.median, p90: ws.p90 } : null,
        wind_measured_sols: [60, 290],
        wind_note: 'on Mars, wind mainly affects dust and visibility, not the astronaut',
        humidity_pct: rh ? rh.median : null,
        dust_optical_depth: dust ? { median: dust.median, p90: dust.p90 } : null,
        sun: sunElev,
        radiation_msv_per_day: rad.dose_equivalent_msv_per_day ?? null,
        radiation_note: 'Curiosity RAD value from Gale crater; no Jezero measurement exists',
        values_are: 'medians of measured data for this local hour, not a forecast'
    };
}

function waypointImagesFor(site, drive) {
    if (!wpImages) return { status: 'not_ready', images: [] };
    const s = Number(site);
    const d = Number(drive);
    const w = wpImages.waypoints.find(x => Number(x.site) === s && Number(x.drive) === d);
    return w ? { status: 'ok', ...w } : { status: 'not_found', images: [] };
}

module.exports = { weatherFor, waypointImagesFor, trekLayers: trek };
