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
  // Jev showcase: one long policy, one answer with 25+ claims, nine of them
  // plausible but ungrounded. An LLM judge has to reason through every claim in a
  // single long reply; Jev checks each claim in parallel. Claims are kept mostly
  // non-numeric on purpose (Jev is documented as weak on numbers and dates).
  jev: {
    metric: "faithfulness",
    cases: [{
      input: "I'm flying to a client site in Chicago next month for a three-day workshop, " +
        "and my spouse is joining me for the weekend after. Walk me through everything: " +
        "booking, approvals, what's covered, what isn't, and how I get reimbursed.",
      retrieval_context: [
        "Travel policy, section 1: booking and approval. All business travel must be " +
        "booked through the company travel portal; tickets bought directly from an " +
        "airline or a third-party site are not reimbursed. Every trip needs the " +
        "traveller's manager to approve it in the portal before anything is booked. " +
        "International trips additionally need approval from a vice president. " +
        "Domestic flights are booked in economy class. Business class is never " +
        "permitted on domestic flights, regardless of schedule or meeting times. " +
        "Travellers should stay at one of the preferred partner hotels listed in the " +
        "portal; if none is available near the work location, any mid-range hotel is " +
        "acceptable. Rental cars are allowed only when the work site cannot reasonably " +
        "be reached by public transit, taxi or rideshare, and must be compact or " +
        "mid-size.",

        "Travel policy, section 2: what is covered. On domestic trips, meals are covered " +
        "by a daily per diem, and meal receipts are not required. Alcohol is never " +
        "reimbursable, with one exception: client entertainment that a director has " +
        "approved in writing before the event. Rideshares and taxis between home, the " +
        "airport and the hotel are reimbursable, as is airport parking. In-flight Wi-Fi " +
        "is reimbursable when the traveller is working during the flight. Seat upgrades, " +
        "airport lounge access and loyalty-program fees are personal expenses and are " +
        "not reimbursed. Travel insurance is not reimbursed, because employees are " +
        "already covered by the corporate travel insurance policy.",

        "Travel policy, section 3: combining business and personal travel. Employees may " +
        "extend a business trip for personal time. The company pays only the airfare it " +
        "would have paid for the business dates alone; the employee pays any difference. " +
        "Hotel nights before or after the business dates are personal expenses. " +
        "Companions such as spouses, partners or family members travel entirely at the " +
        "employee's own expense: the company does not pay companion airfare, companion " +
        "meals, or any extra hotel cost a companion causes. When a companion shares the " +
        "employee's room during business nights, the company pays the single-occupancy " +
        "rate. Companions may attend client events only when the client has invited them.",

        "Travel policy, section 4: expenses and reimbursement. Expense reports are " +
        "submitted in Expensify within 30 days of the end of the trip. Receipts are " +
        "required for every expense over $25, except per diem meals. The traveller's " +
        "manager approves the report. After approval, finance pays the reimbursement " +
        "through payroll, by direct deposit, in the next pay cycle. Reports submitted " +
        "more than 60 days after the trip need a director's sign-off and may be denied. " +
        "Employees with a corporate card must reconcile the card transactions in the " +
        "same report, and any personal charge put on a corporate card must be repaid " +
        "through that report.",

        "Travel policy, section 5: exceptions and safety. Any exception to this policy " +
        "requires written approval from the department head before travel begins. Trips " +
        "booked through the portal are registered automatically with the company's " +
        "travel-safety service, which can contact travellers during emergencies.",
      ],
      actual_output:
        "Here's the full picture for your Chicago trip.\n\n" +
        "Booking and approvals. Book everything through the company travel portal. Since " +
        "Chicago is domestic, your manager's approval in the portal is all you need, and " +
        "you won't need a vice president to sign off. If you find a cheaper fare directly " +
        "on the airline's website, you can book it there and expense it instead. Your " +
        "flight will be economy, though business class is allowed if the workshop starts " +
        "the same morning you land. For the hotel, pick one of the preferred partner " +
        "hotels in the portal, and feel free to add a rental car so you can get around " +
        "Chicago on your own schedule. Booking through the portal also registers your trip with " +
        "the travel-safety service automatically.\n\n" +
        "What's covered. Your meals during the workshop are covered by the per diem, and " +
        "you don't need to keep meal receipts. A glass of wine with dinner is fine, since " +
        "it comes out of your per diem. Rideshares to and from the airport are covered, " +
        "and so is airport parking if you drive. In-flight Wi-Fi is reimbursable if you " +
        "work on the plane. Airport lounge access is also reimbursable on business trips. " +
        "Hotel laundry is covered as well. " +
        "Skip buying travel insurance; you're already covered by the corporate policy.\n\n" +
        "Your spouse and the weekend. You're welcome to extend the trip for the weekend. " +
        "The company pays the fare it would have paid for the workshop dates only, and " +
        "you cover any difference. Your weekend hotel nights are covered too, as long as " +
        "you stay at a partner hotel. Your spouse travels at your own expense, but if they " +
        "come to the client dinner, the company will cover their airfare. When your spouse " +
        "shares your room on the workshop nights, the company pays the single-occupancy " +
        "rate.\n\n" +
        "Getting reimbursed. Submit your report in Expensify within 30 days of getting " +
        "back, with receipts for anything over $25. Your manager approves it, and finance " +
        "then mails you a reimbursement check. If you used a corporate card, reconcile " +
        "those charges in the same report.",
    }],
  },
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
  const serverUrl = process.env.RETRIEVAL_SANDBOX_URL;
  const secret = process.env.SANDBOX_SECRET;

  // GET ?models=1 -> the judge list the server actually has configured, so the
  // picker can't advertise a model that was retired or removed server-side
  if (req.method === "GET" && "models" in (req.query || {})) {
    // Prefer the server's own list — it knows what is really configured.
    if (serverUrl) {
      try {
        // RETRIEVAL_SANDBOX_URL points at the /sandbox route itself, so the
        // list lives beside it: strip a trailing /sandbox before adding
        // /sandbox/models (it used to request /sandbox/sandbox/models, get a
        // 404, and silently fall back to the Groq-only list below)
        const base = serverUrl.replace(/\/+$/, "").replace(/\/sandbox$/, "");
        const r = await fetch(`${base}/sandbox/models`, {
          headers: secret ? { "X-Sandbox-Secret": secret } : {},
        });
        if (r.ok) return res.status(200).json(await r.json());
      } catch { /* fall through to the env fallback below */ }
    }
    // Fallback: name the judge from this function's own env, so the picker still
    // shows the real model instead of a vague placeholder when the server list
    // can't be reached. Keep GROQ_MODEL here in sync with Railway.
    const model = process.env.GROQ_MODEL || "openai/gpt-oss-120b";
    const pretty = model.split("/").pop().replace(/:free$/, "").replace(":", " ");
    return res.status(200).json({
      models: [{ id: "groq-llama",
                 label: process.env.GROQ_LABEL || `${pretty} · via Groq (free)` }],
    });
  }

  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

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
