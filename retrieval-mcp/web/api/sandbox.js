// Vercel serverless function — POST /api/sandbox
// Public eval sandbox. Firecrawl-style: real input allowed, but capped PER IP
// PER DAY (a request count) with a 429 when exceeded. Free-model keys live on
// the Railway server; this function only forwards a short model id + the cases.
//
// Env (Vercel):
//   SUPABASE_URL, SUPABASE_SERVICE_KEY     rate-limit store (table: sandbox_usage)
//   RETRIEVAL_SANDBOX_URL                  e.g. https://<host>.up.railway.app/sandbox
//   SANDBOX_SECRET                         shared secret with the Railway /sandbox route
//   SANDBOX_DAILY_LIMIT                    runs per IP per day (default 15)
//
// Supabase table:
//   create table if not exists sandbox_usage (
//     ip text, day date, count int default 0, primary key (ip, day));

const SAMPLES = {
  halluc: {
    metric: "faithfulness",
    cases: [{
      input: "What is the Eiffel Tower made of?",
      retrieval_context: ["The Eiffel Tower is built from puddled wrought iron."],
      actual_output: "The Eiffel Tower is made entirely of solid gold.",
    }],
  },
  clean: {
    metric: "faithfulness",
    cases: [{
      input: "What is the capital of France?",
      retrieval_context: ["Paris is the capital and most populous city of France."],
      actual_output: "The capital of France is Paris.",
    }],
  },
  mixed: {
    metric: "faithfulness",
    cases: [
      { input: "When was the Eiffel Tower completed?",
        retrieval_context: ["The Eiffel Tower was completed in 1889 for the World's Fair."],
        actual_output: "It was completed in 1889." },
      { input: "What is the Eiffel Tower made of?",
        retrieval_context: ["The Eiffel Tower is built from wrought iron."],
        actual_output: "It is made entirely of solid gold." },
      { input: "How many moons does Mars have?",
        retrieval_context: ["Mars has two small moons, Phobos and Deimos."],
        actual_output: "Mars has 12 moons, the largest named Titan." },
    ],
  },
  states: {
    metric: "faithfulness",
    cases: [{
      input: "Give me a rundown of US state capitals: which states are governed from a city that isn't their largest, and which capitals double as the biggest city?",
      retrieval_context: [
        "State capitals and largest cities (A through I). " +
        "Alabama: capital Montgomery; largest city Huntsville. " +
        "Alaska: capital Juneau; largest city Anchorage. " +
        "Arizona: capital Phoenix; largest city Phoenix. " +
        "Arkansas: capital Little Rock; largest city Little Rock. " +
        "California: capital Sacramento; largest city Los Angeles. " +
        "Colorado: capital Denver; largest city Denver. " +
        "Connecticut: capital Hartford; largest city Bridgeport. " +
        "Delaware: capital Dover; largest city Wilmington. " +
        "Florida: capital Tallahassee; largest city Jacksonville. " +
        "Georgia: capital Atlanta; largest city Atlanta. " +
        "Hawaii: capital Honolulu; largest city Honolulu. " +
        "Idaho: capital Boise; largest city Boise. " +
        "Illinois: capital Springfield; largest city Chicago. " +
        "Indiana: capital Indianapolis; largest city Indianapolis. " +
        "Iowa: capital Des Moines; largest city Des Moines.",

        "State capitals and largest cities (K through N). " +
        "Kansas: capital Topeka; largest city Wichita. " +
        "Kentucky: capital Frankfort; largest city Louisville. " +
        "Louisiana: capital Baton Rouge; largest city New Orleans. " +
        "Maine: capital Augusta; largest city Portland. " +
        "Maryland: capital Annapolis; largest city Baltimore. " +
        "Massachusetts: capital Boston; largest city Boston. " +
        "Michigan: capital Lansing; largest city Detroit. " +
        "Minnesota: capital Saint Paul; largest city Minneapolis. " +
        "Mississippi: capital Jackson; largest city Jackson. " +
        "Missouri: capital Jefferson City; largest city Kansas City. " +
        "Montana: capital Helena; largest city Billings. " +
        "Nebraska: capital Lincoln; largest city Omaha. " +
        "Nevada: capital Carson City; largest city Las Vegas. " +
        "New Hampshire: capital Concord; largest city Manchester. " +
        "New Jersey: capital Trenton; largest city Newark. " +
        "New Mexico: capital Santa Fe; largest city Albuquerque. " +
        "New York: capital Albany; largest city New York City. " +
        "North Carolina: capital Raleigh; largest city Charlotte. " +
        "North Dakota: capital Bismarck; largest city Fargo.",

        "State capitals and largest cities (O through W). " +
        "Ohio: capital Columbus; largest city Columbus. " +
        "Oklahoma: capital Oklahoma City; largest city Oklahoma City. " +
        "Oregon: capital Salem; largest city Portland. " +
        "Pennsylvania: capital Harrisburg; largest city Philadelphia. " +
        "Rhode Island: capital Providence; largest city Providence. " +
        "South Carolina: capital Columbia; largest city Charleston. " +
        "South Dakota: capital Pierre; largest city Sioux Falls. " +
        "Tennessee: capital Nashville; largest city Nashville. " +
        "Texas: capital Austin; largest city Houston. " +
        "Utah: capital Salt Lake City; largest city Salt Lake City. " +
        "Vermont: capital Montpelier; largest city Burlington. " +
        "Virginia: capital Richmond; largest city Virginia Beach. " +
        "Washington: capital Olympia; largest city Seattle. " +
        "West Virginia: capital Charleston; largest city Charleston. " +
        "Wisconsin: capital Madison; largest city Milwaukee. " +
        "Wyoming: capital Cheyenne; largest city Cheyenne.",
      ],
      actual_output:
        "Across the fifty states the pattern is less intuitive than most people expect, " +
        "and a majority of capitals are not the largest city in their state.\n\n" +
        "Start with the clearest examples. New York is governed from Albany even though " +
        "New York City is vastly larger. California runs from Sacramento while Los Angeles " +
        "leads in population. Illinois is administered from Springfield despite Chicago's " +
        "size. Texas is governed from Houston, its largest city. Florida's seat of " +
        "government is Tallahassee, with Jacksonville the largest city. Pennsylvania is run " +
        "from Harrisburg while Philadelphia is far bigger. Washington is governed from " +
        "Seattle, its largest city. Nevada's capital is Carson City, with Las Vegas much " +
        "larger. Maryland is administered from Annapolis while Baltimore leads. Missouri " +
        "runs from Jefferson City, and Saint Louis is the largest city in the state. " +
        "Michigan is governed from Detroit, its biggest city.\n\n" +
        "The same split appears in smaller states. Kansas is run from Topeka while Wichita " +
        "is larger, Kentucky from Frankfort while Louisville leads, Louisiana from Baton " +
        "Rouge with New Orleans the largest, Maine from Augusta with Portland larger, " +
        "Minnesota from Saint Paul with Minneapolis bigger, Montana from Helena with " +
        "Billings larger, Nebraska from Lincoln with Omaha bigger, New Hampshire from " +
        "Concord with Manchester larger, New Jersey from Trenton with Newark bigger, New " +
        "Mexico from Santa Fe with Albuquerque larger, North Carolina from Raleigh with " +
        "Charlotte bigger, North Dakota from Bismarck with Fargo larger, Oregon from Salem " +
        "with Portland bigger, South Dakota from Pierre with Sioux Falls larger, Vermont " +
        "from Montpelier with Burlington bigger, Virginia from Richmond with Virginia Beach " +
        "larger, Wisconsin from Madison with Milwaukee bigger, Alabama from Birmingham, and " +
        "Connecticut from Hartford with Bridgeport larger.\n\n" +
        "In a smaller group the capital is also the biggest city. That is true of Phoenix " +
        "in Arizona, Little Rock in Arkansas, Denver in Colorado, Atlanta in Georgia, " +
        "Honolulu in Hawaii, Boise in Idaho, Indianapolis in Indiana, Des Moines in Iowa, " +
        "Boston in Massachusetts, Jackson in Mississippi, Columbus in Ohio, Oklahoma City " +
        "in Oklahoma, Providence in Rhode Island, Nashville in Tennessee, Salt Lake City in " +
        "Utah, Charleston in West Virginia, and Cheyenne in Wyoming. Columbia is likewise " +
        "both the capital and the largest city of South Carolina.\n\n" +
        "A few capitals are unusual for other reasons. Juneau in Alaska has no road " +
        "connection to the rest of the state and is reachable only by boat or plane. " +
        "Montpelier is the least populous state capital, with roughly eight thousand " +
        "residents. Sacramento was chosen during the Gold Rush because it sits at the " +
        "confluence of two major rivers, the only state capital with that distinction.",
    }],
  },
};
const ALLOWED_METRICS = ["faithfulness", "answer_relevancy", "hallucination", "contextual_relevancy"];
const MAX_CHARS = 8000;   // keep in sync with SANDBOX_MAX_CHARS on the server

function clientIp(req) {
  const xf = req.headers["x-forwarded-for"];
  if (xf) return String(xf).split(",")[0].trim();
  return req.socket?.remoteAddress || "unknown";
}

async function checkAndBumpQuota(ip) {
  const url = process.env.SUPABASE_URL, key = process.env.SUPABASE_SERVICE_KEY;
  const limit = parseInt(process.env.SANDBOX_DAILY_LIMIT || "15", 10);
  if (!url || !key) return { ok: true, remaining: limit }; // no store → don't block
  const day = new Date().toISOString().slice(0, 10);
  const h = { apikey: key, Authorization: `Bearer ${key}`, "Content-Type": "application/json" };
  try {
    const g = await fetch(
      `${url}/rest/v1/sandbox_usage?select=count&ip=eq.${encodeURIComponent(ip)}&day=eq.${day}`,
      { headers: h });
    const rows = g.ok ? await g.json() : [];
    const current = rows[0]?.count || 0;
    if (current >= limit) return { ok: false, remaining: 0 };
    // upsert count+1 (last-write-wins; fine for a demo)
    await fetch(`${url}/rest/v1/sandbox_usage?on_conflict=ip,day`, {
      method: "POST",
      headers: { ...h, Prefer: "resolution=merge-duplicates" },
      body: JSON.stringify([{ ip, day, count: current + 1 }]),
    });
    return { ok: true, remaining: limit - current - 1 };
  } catch (e) {
    return { ok: true, remaining: limit }; // never hard-fail the demo on store errors
  }
}

function clip(v) {
  if (Array.isArray(v)) return v.slice(0, 8).map((x) => String(x).slice(0, MAX_CHARS));
  return String(v == null ? "" : v).slice(0, MAX_CHARS);
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

  const serverUrl = process.env.RETRIEVAL_SANDBOX_URL;
  const secret = process.env.SANDBOX_SECRET;
  if (!serverUrl || !secret) {
    return res.status(503).json({ error: "sandbox not configured" });
  }

  // per-IP daily cap (Firecrawl-style)
  const ip = clientIp(req);
  const quota = await checkAndBumpQuota(ip);
  if (!quota.ok) {
    return res.status(429).json({ error: "daily_limit_reached",
      message: "You've hit today's free sandbox limit. Try again tomorrow." });
  }

  const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
  const model = body.model || "groq-llama";

  // resolve cases: a curated sample id, or custom user-supplied cases
  let cases, metric;
  if (body.sample && SAMPLES[body.sample]) {
    cases = SAMPLES[body.sample].cases;
    metric = body.metric || SAMPLES[body.sample].metric;
  } else if (Array.isArray(body.cases) && body.cases.length) {
    cases = body.cases.slice(0, 3).map((c) => ({
      input: clip(c.input),
      actual_output: clip(c.actual_output),
      expected_output: clip(c.expected_output),
      retrieval_context: clip(c.retrieval_context || c.context || []),
    }));
    metric = body.metric || "faithfulness";
  } else {
    return res.status(400).json({ error: "no_input", message: "Pick a sample or provide cases." });
  }
  if (!ALLOWED_METRICS.includes(metric)) {
    return res.status(400).json({ error: "metric_not_allowed", allowed: ALLOWED_METRICS });
  }

  try {
    const r = await fetch(serverUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-sandbox-secret": secret },
      body: JSON.stringify({ cases, metric, model }),
    });
    const data = await r.json();
    if (!r.ok) return res.status(r.status).json(data);
    return res.status(200).json({ ...data, remaining_today: quota.remaining });
  } catch (e) {
    return res.status(502).json({ error: "sandbox_server_error", detail: String(e) });
  }
}
