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
  // Jev: a made-up board game. The answer explains the rules with 6 wrong:
  // 4 contradict the rulebook (who starts, repeating actions, the storm, trading
  // with players) and 2 are invented (the lighthouse card, the Crown Isle bonus).
  jev_game: {
    metric: "faithfulness",
    cases: [{
      input: "It's our first time playing Lanterns of Vell tonight. Explain how a turn works, how you win, what the storm does, and anything a new player usually gets wrong.",
      retrieval_context: ["Lanterns of Vell rulebook, part 1: setup. Each player starts with three lantern tokens and five coins. The harbor board is placed in the middle, and the storm marker starts on the calm space. The youngest player goes first, and play continues clockwise. Part 2: a turn. On your turn you take exactly two actions. You may sail one space, trade at the market, or light a lantern on an island you are docked at. You may take the same action twice. Lighting a lantern costs two coins. At the end of your turn, draw one tide card and resolve it. Part 3: winning. The first player to light lanterns on four different islands wins immediately. If the tide deck runs out before anyone wins, the player with the most lit lanterns wins, and ties go to the player with more coins. Part 4: the storm. When a tide card shows a lightning bolt, move the storm marker one space along the storm track. When the storm reaches the tempest space, every player loses one lantern token that has not been lit yet, and the storm marker returns to the calm space. Lit lanterns are never lost to the storm. Part 5: trading. At the market you may exchange two coins for one lantern token, or one lantern token for three coins. You cannot trade with other players. Part 6: common mistakes. Players often forget to draw a tide card at the end of their turn. Sailing onto an island does not light it; lighting is a separate action."],
      actual_output: "Welcome to Lanterns of Vell! Everyone starts with three lantern tokens and five coins, the storm marker begins on the calm space, and the oldest player goes first. On your turn you take exactly two actions: sail one space, trade at the market, or light a lantern on an island where you're docked. You can't take the same action twice in one turn, so plan your pair carefully. Lighting a lantern costs two coins. At the end of your turn you draw one tide card and resolve it. To win, be the first to light lanterns on four different islands. If the tide deck runs out first, whoever has the most lit lanterns wins, and ties go to whoever has more coins. The storm is the tricky part. Each lightning bolt on a tide card moves the storm marker one space. When it reaches the tempest space, every player loses one lantern, lit or unlit, so don't get attached to your early islands. The marker then returns to the calm space. At the market you can swap two coins for a lantern token or a lantern token for three coins, and you can also trade directly with other players if you both agree. Two tips: players often forget to draw a tide card at the end of their turn, and sailing onto an island doesn't light it, since lighting is its own action. Also, the lighthouse card lets you light any island for free once per game, and the first player to light the Crown Isle gets a bonus of three coins.",
    }],
  },
  // Jev: a long API rundown, ~50 claims, 15 wrong (5 of them changed numbers,
  // Jev's documented weak spot). Measured: Jev caught 15 of 15, no false alarms.
  jev_api: {
    metric: "faithfulness",
    cases: [{
      input: "We're integrating Harbor into our billing system and moving to production next week. Give me a complete rundown: authentication, rate limits, webhooks, pagination and errors, data retention, and what uptime and support we get on our Starter plan.",
      retrieval_context: ["Harbor API documentation, section 1: authentication. Every request must include an API key in the Authorization header as a bearer token. Keys are created in the dashboard under Settings, then API keys. Each key belongs to one workspace and cannot be shared across workspaces. Keys can be restricted to read-only access. Secret keys start with the prefix hb_live for production and hb_test for the sandbox environment. A revoked key stops working immediately. Harbor never shows a secret key again after it is created; if a key is lost, create a new one and revoke the old one.\n\nSection 2: rate limits. Production keys may send up to one hundred requests per second per workspace. Sandbox keys are limited to ten requests per second. When a limit is exceeded, the API returns status 429 with a Retry-After header giving the number of seconds to wait. Requests above the limit are not queued; the excess requests are rejected. Enterprise plans can request a higher limit through support.\n\nSection 3: webhooks. Harbor sends webhooks as HTTPS POST requests with a JSON body. Each webhook includes a Harbor-Signature header containing an HMAC-SHA256 of the raw request body, computed with the endpoint's signing secret. Receivers should verify the signature before trusting the payload. Endpoints must respond with a 2xx status within ten seconds, otherwise the delivery counts as failed. Failed deliveries are retried with exponential backoff for up to three days. Events may arrive out of order and may be delivered more than once, so receivers should deduplicate using the event id. Webhook endpoints that fail continuously for three days are disabled automatically, and the workspace owner is notified by email.\n\nSection 4: pagination and errors. List endpoints use cursor-based pagination. Each response includes a next_cursor field; pass it as the cursor parameter to fetch the next page. The default page size is twenty-five and the maximum page size is one hundred. Errors return a JSON body with a type, a code and a human-readable message. Validation errors use status 422. Requests that are safe to retry can include an Idempotency-Key header, and Harbor stores idempotency keys for twenty-four hours.\n\nSection 5: data retention and deletion. Event logs are kept for thirty days on the Starter plan and for one year on the Enterprise plan. Deleted records are removed from the primary database immediately and from backups within thirty-five days. Customers can export all workspace data in JSON format from the dashboard at any time. Harbor stores data in the United States by default; an EU data region is available on Enterprise plans only.\n\nSection 6: service level and support. Enterprise plans include a monthly uptime commitment of 99.95 percent. If uptime falls below the commitment, customers receive service credits, which are the sole remedy for downtime. Starter plans have no uptime commitment. Support for Starter plans is by email, with a response within two business days. Enterprise support includes a dedicated channel with a one-hour response for critical issues. Harbor announces breaking API changes at least ninety days in advance, and old API versions remain available for twelve months after a new version is released."],
      actual_output: "Here's a complete rundown for taking Harbor to production.\n\nAuthentication. Every request needs an API key sent in the Authorization header as a bearer token. You create keys in the dashboard under Settings, then API keys. A key belongs to one workspace, but you can share it across your other workspaces if you want a single key for everything. You can restrict a key to read-only access, which is a good idea for reporting jobs. Production secret keys start with hb_live and sandbox keys start with hb_test, so make sure your deployment uses the hb_live key. Revoking a key takes effect immediately. If you lose a secret key, you can view it again in the dashboard under Settings.\n\nRate limits. Production keys can send up to one hundred requests per second per workspace. Sandbox keys are limited to fifty requests per second, so load tests in the sandbox will behave much like production. If you go over the limit, the API returns a 429 with a Retry-After header telling you how many seconds to wait. Requests above the limit are queued and processed once capacity frees up, so short bursts are safe. If you outgrow the limit, Enterprise plans can request a higher one through support.\n\nWebhooks. Harbor delivers webhooks as HTTPS POST requests with a JSON body. Each one carries a Harbor-Signature header, an HMAC-SHA1 of the raw body computed with your endpoint's signing secret. Always verify that signature before you trust the payload. Your endpoint has to return a 2xx status within ten seconds or the delivery counts as failed. Failed deliveries are retried with exponential backoff for up to seven days. Events always arrive in the order they happened, but the same event can be delivered more than once, so deduplicate using the event id. If your endpoint keeps failing for three days, Harbor disables it automatically and emails the workspace owner.\n\nPagination and errors. List endpoints use cursor-based pagination: each response has a next_cursor field, and you pass it back as the cursor parameter to get the next page. The default page size is twenty-five, and you can raise it to a maximum of five hundred per page. Errors come back as JSON with a type, a code and a human-readable message. Validation errors use status 422. For requests you might retry, send an Idempotency-Key header; Harbor keeps idempotency keys for seven days, so retries within a week are safe.\n\nData retention. On the Starter plan, event logs are kept for one year, the same as Enterprise. When you delete a record, it's removed from the primary database immediately and from backups within thirty-five days. You can export all of your workspace data as JSON from the dashboard at any time. Data is stored in the United States by default, and Starter customers can request the EU data region through support.\n\nUptime and support on Starter. Your Starter plan includes a monthly uptime commitment of 99.9 percent. If Harbor misses it, you can claim a full refund for the affected month. Support on Starter is by email, with a response within two business days; the dedicated channel with one-hour responses is Enterprise only. Harbor announces breaking API changes at least ninety days in advance, and old API versions stay available for twelve months after a new version ships.\n\nA few extras worth knowing: Harbor is SOC 2 Type II certified, which should help with your security review, and there are official SDKs for Python, Node and Go that handle signatures and pagination for you. Good luck with the launch.",
    }],
  },
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
