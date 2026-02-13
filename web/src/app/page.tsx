"use client";

import { useMemo, useState } from "react";

type FormState = {
  ageBand: string;
  bmi: number;
  smoker: string;
  activity: string;
  sleep: number;
  genHealth: string;
  alcohol: string;
  diffWalking: string;
};

type DiseaseMeta = {
  id: string;
  name: string;
  icon: string;
  baseRisk: number;
  model: string;
  accuracy: string;
  factor: number;
};

type ToggleKey = "smoker" | "activity" | "alcohol" | "diffWalking";

const diseaseMeta: DiseaseMeta[] = [
  {
    id: "heart",
    name: "Heart Disease",
    icon: "❤️",
    baseRisk: 46.34,
    model: "Random Forest",
    accuracy: "72.98%",
    factor: 1.15,
  },
  {
    id: "stroke",
    name: "Stroke",
    icon: "🧠",
    baseRisk: 8.8,
    model: "Decision Tree",
    accuracy: "91.14%",
    factor: 1.3,
  },
  {
    id: "diabetes",
    name: "Diabetes",
    icon: "🩸",
    baseRisk: 21.29,
    model: "Random Forest",
    accuracy: "80.39%",
    factor: 1.2,
  },
  {
    id: "asthma",
    name: "Asthma",
    icon: "🫁",
    baseRisk: 15.69,
    model: "Random Forest",
    accuracy: "85.08%",
    factor: 0.9,
  },
  {
    id: "kidney",
    name: "Kidney Disease",
    icon: "🫘",
    baseRisk: 7.44,
    model: "Logistic Regression",
    accuracy: "92.41%",
    factor: 1.25,
  },
  {
    id: "skin",
    name: "Skin Cancer",
    icon: "🧬",
    baseRisk: 13.77,
    model: "Logistic Regression",
    accuracy: "86.49%",
    factor: 0.8,
  },
];

const ageWeights: Record<string, number> = {
  "18-24": 0.1,
  "25-34": 0.2,
  "35-44": 0.4,
  "45-54": 0.6,
  "55-64": 0.9,
  "65-74": 1.1,
  "75+": 1.3,
};

const healthScore: Record<string, number> = {
  Excellent: -0.6,
  "Very good": -0.3,
  Good: 0.1,
  Fair: 0.5,
  Poor: 0.9,
};

const labelPill =
  "rounded-full border border-black/10 bg-white/80 px-4 py-1 text-xs font-semibold uppercase tracking-[0.25em] text-[color:var(--cool)]";

const cardClass =
  "rounded-3xl border border-black/10 bg-[color:var(--panel)] shadow-[0_20px_45px_rgba(0,0,0,0.08)]";

const toggleFields: { label: string; key: ToggleKey }[] = [
  { label: "Smoker", key: "smoker" },
  { label: "Physical activity", key: "activity" },
  { label: "Alcohol", key: "alcohol" },
  { label: "Difficulty walking", key: "diffWalking" },
];

function clampRisk(value: number) {
  return Math.min(98, Math.max(2, value));
}

function computeRisk(form: FormState) {
  const age = ageWeights[form.ageBand] ?? 0.3;
  const bmiAdj = form.bmi >= 30 ? 0.9 : form.bmi >= 25 ? 0.5 : 0.1;
  const smokeAdj = form.smoker === "Yes" ? 1.0 : 0;
  const activityAdj = form.activity === "Yes" ? -0.4 : 0.4;
  const sleepAdj = form.sleep < 6 ? 0.5 : form.sleep > 8 ? 0.3 : 0.1;
  const genHealthAdj = healthScore[form.genHealth] ?? 0.2;
  const alcoholAdj = form.alcohol === "Yes" ? 0.3 : 0;
  const walkingAdj = form.diffWalking === "Yes" ? 0.6 : 0;

  const composite =
    age + bmiAdj + smokeAdj + activityAdj + sleepAdj + genHealthAdj + alcoholAdj + walkingAdj;

  return diseaseMeta.map((disease) => {
    const risk = clampRisk(disease.baseRisk + composite * disease.factor * 9);
    return {
      ...disease,
      risk,
      band:
        risk >= 65 ? "High" : risk >= 40 ? "Moderate" : "Low",
    };
  });
}

export default function Home() {
  const [form, setForm] = useState<FormState>({
    ageBand: "45-54",
    bmi: 25.5,
    smoker: "No",
    activity: "Yes",
    sleep: 7,
    genHealth: "Good",
    alcohol: "No",
    diffWalking: "No",
  });
  const [hasRun, setHasRun] = useState(false);

  const results = useMemo(() => computeRisk(form), [form]);
  const topRisk = results.reduce((max, item) => (item.risk > max.risk ? item : max), results[0]);

  return (
    <div className="min-h-screen px-6 pb-20 pt-10 text-[color:var(--foreground)]">
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-16">
        <header className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
          <div className="flex flex-col gap-6">
            <span className={labelPill}>Predictive Health Lab</span>
            <h1 className="display-font text-4xl font-semibold leading-tight text-[color:var(--foreground)] sm:text-5xl lg:text-6xl">
              PredictaMed delivers a cinematic health risk explorer for six chronic conditions.
            </h1>
            <p className="max-w-xl text-lg leading-relaxed text-[color:var(--muted)]">
              A high-fidelity UI that mirrors the research pipeline: patient context, vital
              lifestyle signals, and model performance snapshots. Built for portfolios, demos,
              and stakeholder walkthroughs.
            </p>
            <div className="flex flex-wrap gap-4">
              <button
                className="rounded-full bg-[color:var(--accent)] px-7 py-3 text-sm font-semibold uppercase tracking-[0.2em] text-white shadow-[0_12px_24px_rgba(255,122,69,0.35)] transition hover:-translate-y-0.5"
                onClick={() => setHasRun(true)}
              >
                Run Demo
              </button>
              <a
                className="rounded-full border border-black/10 bg-white/70 px-7 py-3 text-sm font-semibold uppercase tracking-[0.2em] text-[color:var(--foreground)] backdrop-blur transition hover:border-black/30"
                href="https://github.com/ZayyanCPU/PredictaMed"
                target="_blank"
                rel="noreferrer"
              >
                View Repo
              </a>
            </div>
          </div>
          <div className={`${cardClass} relative overflow-hidden p-8`}>
            <div className="absolute -right-12 -top-10 h-40 w-40 rounded-full bg-[color:var(--cool-soft)] opacity-80" />
            <div className="absolute -bottom-16 -left-12 h-48 w-48 rounded-full bg-[color:var(--highlight)] opacity-40" />
            <div className="relative z-10 flex flex-col gap-6">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.3em] text-[color:var(--cool)]">
                  Data Snapshot
                </p>
                <h2 className="display-font mt-2 text-3xl">59,068 patient records</h2>
                <p className="text-sm text-[color:var(--muted)]">
                  18 clinical and behavioral attributes • 6 disease models
                </p>
              </div>
              <div className="grid gap-4 sm:grid-cols-3">
                {[
                  { label: "Model Suite", value: "6 algorithms" },
                  { label: "Signals", value: "BMI, sleep, activity" },
                  { label: "Validation", value: "AUC + Accuracy" },
                ].map((item) => (
                  <div key={item.label} className="rounded-2xl bg-white/80 p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--cool)]">
                      {item.label}
                    </p>
                    <p className="mt-2 text-sm font-semibold text-[color:var(--foreground)]">
                      {item.value}
                    </p>
                  </div>
                ))}
              </div>
              <p className="text-xs uppercase tracking-[0.25em] text-[color:var(--muted)]">
                Educational UI — not medical advice
              </p>
            </div>
          </div>
        </header>

        <section className="grid gap-6 lg:grid-cols-3">
          {[
            {
              title: "Multi-disease workflow",
              text: "Track heart disease, stroke, diabetes, asthma, kidney disease, and skin cancer in a single narrative dashboard.",
            },
            {
              title: "Clinician-ready visuals",
              text: "A layered interface with story cards, color-coded risk bands, and model performance summaries.",
            },
            {
              title: "Explainable insights",
              text: "Surface key drivers like BMI, sleep, activity, and perceived health for clear stakeholder alignment.",
            },
          ].map((item) => (
            <div key={item.title} className={`${cardClass} p-6`}> 
              <h3 className="display-font text-2xl">{item.title}</h3>
              <p className="mt-3 text-sm leading-relaxed text-[color:var(--muted)]">{item.text}</p>
            </div>
          ))}
        </section>

        <section className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr]">
          <div className={`${cardClass} p-8`}>
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.3em] text-[color:var(--cool)]">
                  Risk Explorer
                </p>
                <h2 className="display-font mt-2 text-3xl">Personalized risk preview</h2>
              </div>
              <span className="rounded-full bg-[color:var(--cool-soft)] px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--cool)]">
                Demo mode
              </span>
            </div>

            <div className="mt-8 grid gap-6 sm:grid-cols-2">
              <label className="flex flex-col gap-2 text-sm font-semibold">
                Age band
                <select
                  className="rounded-2xl border border-black/10 bg-white/80 px-4 py-3 text-sm"
                  value={form.ageBand}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, ageBand: event.target.value }))
                  }
                >
                  {Object.keys(ageWeights).map((age) => (
                    <option key={age} value={age}>
                      {age}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-2 text-sm font-semibold">
                General health
                <select
                  className="rounded-2xl border border-black/10 bg-white/80 px-4 py-3 text-sm"
                  value={form.genHealth}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, genHealth: event.target.value }))
                  }
                >
                  {Object.keys(healthScore).map((level) => (
                    <option key={level} value={level}>
                      {level}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-3 text-sm font-semibold">
                BMI: <span className="text-[color:var(--accent-strong)]">{form.bmi.toFixed(1)}</span>
                <input
                  type="range"
                  min={16}
                  max={45}
                  step={0.1}
                  value={form.bmi}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, bmi: Number(event.target.value) }))
                  }
                />
              </label>
              <label className="flex flex-col gap-3 text-sm font-semibold">
                Sleep (hours): <span className="text-[color:var(--accent-strong)]">{form.sleep}</span>
                <input
                  type="range"
                  min={4}
                  max={10}
                  step={1}
                  value={form.sleep}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, sleep: Number(event.target.value) }))
                  }
                />
              </label>
            </div>

            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              {toggleFields.map((item) => (
                <div key={item.key} className="flex items-center justify-between gap-3">
                  <span className="text-sm font-semibold">{item.label}</span>
                  <div className="flex gap-2">
                    {["No", "Yes"].map((value) => (
                      <button
                        key={value}
                        className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] transition ${
                          form[item.key] === value
                            ? "bg-[color:var(--accent)] text-white"
                            : "border border-black/10 bg-white/70 text-[color:var(--muted)]"
                        }`}
                        onClick={() =>
                          setForm((prev) => ({
                            ...prev,
                            [item.key]: value,
                          }))
                        }
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <button
              className="mt-8 w-full rounded-full bg-[color:var(--accent)] px-6 py-4 text-sm font-semibold uppercase tracking-[0.25em] text-white shadow-[0_12px_24px_rgba(255,122,69,0.35)] transition hover:-translate-y-0.5"
              onClick={() => setHasRun(true)}
            >
              Analyze Risk Profile
            </button>
            <p className="mt-4 text-xs text-[color:var(--muted)]">
              Demo prediction engine. Outputs are illustrative, not medical guidance.
            </p>
          </div>

          <div className={`${cardClass} p-8`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.3em] text-[color:var(--cool)]">
                  Results
                </p>
                <h3 className="display-font mt-2 text-3xl">{hasRun ? "Risk bands" : "Awaiting input"}</h3>
              </div>
              <span className="rounded-full bg-[color:var(--highlight)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--foreground)]">
                {hasRun ? `${topRisk.name} leading` : "Run demo"}
              </span>
            </div>

            <div className="mt-6 space-y-4">
              {results.map((item) => (
                <div key={item.id} className="rounded-2xl border border-black/10 bg-white/80 p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-xl">{item.icon}</span>
                      <div>
                        <p className="text-sm font-semibold">{item.name}</p>
                        <p className="text-xs text-[color:var(--muted)]">Best model: {item.model}</p>
                      </div>
                    </div>
                    <span
                      className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] ${
                        item.band === "High"
                          ? "bg-red-100 text-red-600"
                          : item.band === "Moderate"
                          ? "bg-amber-100 text-amber-700"
                          : "bg-emerald-100 text-emerald-700"
                      }`}
                    >
                      {item.band}
                    </span>
                  </div>
                  <div className="mt-4">
                    <div className="flex items-center justify-between text-xs text-[color:var(--muted)]">
                      <span>Estimated probability</span>
                      <span className="text-sm font-semibold text-[color:var(--foreground)]">
                        {item.risk.toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-black/5">
                      <div
                        className="h-full rounded-full bg-[color:var(--accent)]"
                        style={{ width: `${item.risk}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-2xl bg-[color:var(--cool-soft)] p-4 text-xs text-[color:var(--cool)]">
              Highlight: {topRisk.name} at {topRisk.risk.toFixed(1)}% estimated probability.
            </div>
          </div>
        </section>

        <section className={`${cardClass} p-8`}>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.3em] text-[color:var(--cool)]">
                Model Performance
              </p>
              <h3 className="display-font mt-2 text-3xl">Best-in-class validation results</h3>
            </div>
            <span className="rounded-full bg-white/70 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--muted)]">
              Accuracy + AUC
            </span>
          </div>
          <div className="mt-6 grid gap-4 md:grid-cols-3">
            {diseaseMeta.map((disease) => (
              <div key={disease.id} className="rounded-2xl border border-black/10 bg-white/80 p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-[color:var(--cool)]">
                  {disease.name}
                </p>
                <p className="mt-2 text-lg font-semibold text-[color:var(--foreground)]">
                  {disease.model}
                </p>
                <p className="text-sm text-[color:var(--muted)]">Accuracy {disease.accuracy}</p>
              </div>
            ))}
          </div>
        </section>

        <footer className="flex flex-col gap-4 text-sm text-[color:var(--muted)]">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <p>PredictaMed • AI-powered multi-disease prediction UI</p>
            <a
              className="text-[color:var(--foreground)] underline decoration-[color:var(--accent)] decoration-2 underline-offset-4"
              href="mailto:zayyanahmad765@gmail.com"
            >
              Contact
            </a>
          </div>
          <p>
            This interface is a visual concept for educational purposes and does not provide
            medical diagnosis.
          </p>
        </footer>
      </main>
    </div>
  );
}
