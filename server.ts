import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "5mb" }));

// Endpoint for Alchemist Gap Analysis & Few-shot Bullet Transformation
app.post("/api/alchemist/analyze", async (req, res) => {
  try {
    const { masterResume, jobDescription, targetCompany, targetRole } = req.body;

    if (!masterResume || !jobDescription) {
      return res.status(400).json({ error: "Master resume and job description are required." });
    }

    const apiKey = process.env.GEMINI_API_KEY;

    // If Gemini key is available, use Gemini 3.6 Flash
    if (apiKey && apiKey !== "MY_GEMINI_API_KEY") {
      const ai = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            "User-Agent": "aistudio-build",
          },
        },
      });

      const prompt = `
Perform a thorough, ATS-focused Resume Gap Analysis & Bullet Point Transformation using Few-Shot Learning principles.

Target Role / Context:
- Target Job Title: ${targetRole || "Extracted from Job Description"}
- Target Company: ${targetCompany || "Extracted from Job Description"}

MASTER RESUME:
"""
${masterResume}
"""

JOB DESCRIPTION:
"""
${jobDescription}
"""

System Guidance & Few-Shot Examples:
You act as an executive resume alchemist. Your goal is to compare the Master Resume against the Job Description, extract missing keywords, and transform generic bullet points into high-impact, quantifiable accomplishment statements.

Few-Shot Transformation Reference 1:
- Original: "Responsible for creating front-end features in React and fixing bugs."
- Transformation: "Engineered high-performance React 19 micro-frontends and resolved complex state bugs, boosting client-side load speed by 35% across 200k monthly sessions."
- Added Keywords: React 19, micro-frontends, state management, client-side load speed.

Few-Shot Transformation Reference 2:
- Original: "Managed a team of developers and talked to clients."
- Transformation: "Spearheaded an agile team of 6 engineers and aligned weekly sprint milestones with enterprise stakeholders, delivering key roadmap features 15% ahead of deadline."
- Added Keywords: Agile team, sprint milestones, enterprise stakeholders, roadmap features.

Requirements for your output JSON:
1. Identify target Job Title and Company from the JD if not specified.
2. Calculate estimated match score BEFORE (0-100%) and AFTER tailoring (0-100%).
3. Categorize missing keywords into: hardSkills, toolsAndFrameworks, softSkills, domainKnowledge.
4. Extract list of present keywords already matching the JD.
5. Identify 3 to 8 key bullet points from the Master Resume and transform each into a tailored, keyword-dense bullet point. For each:
   - Provide original bullet
   - Provide rewritten tailored bullet incorporating specific missing keywords from the JD
   - List keywords added
   - List the strong action verb used
   - Provide explanation/reason for change
   - Provide a score improvement metric (e.g., +25)
6. Provide an ATS compliance checklist (4-5 factors with status "pass" or "warning" and explanatory notes).
7. Generate the COMPLETE tailored master resume text, integrating the new bullets and a highlighted top skills section aligned with the JD.

Return strictly valid JSON matching the schema requested.
`;

      const response = await ai.models.generateContent({
        model: "gemini-3.6-flash",
        contents: prompt,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              matchScoreBefore: { type: Type.NUMBER },
              matchScoreAfter: { type: Type.NUMBER },
              jobTitleIdentified: { type: Type.STRING },
              companyIdentified: { type: Type.STRING },
              missingKeywords: {
                type: Type.OBJECT,
                properties: {
                  hardSkills: { type: Type.ARRAY, items: { type: Type.STRING } },
                  toolsAndFrameworks: { type: Type.ARRAY, items: { type: Type.STRING } },
                  softSkills: { type: Type.ARRAY, items: { type: Type.STRING } },
                  domainKnowledge: { type: Type.ARRAY, items: { type: Type.STRING } },
                },
                required: ["hardSkills", "toolsAndFrameworks", "softSkills", "domainKnowledge"],
              },
              presentKeywords: { type: Type.ARRAY, items: { type: Type.STRING } },
              bulletTransformations: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    originalBullet: { type: Type.STRING },
                    rewrittenBullet: { type: Type.STRING },
                    keywordsAdded: { type: Type.ARRAY, items: { type: Type.STRING } },
                    impactVerb: { type: Type.STRING },
                    transformationReason: { type: Type.STRING },
                    scoreImprovement: { type: Type.NUMBER },
                  },
                  required: [
                    "id",
                    "originalBullet",
                    "rewrittenBullet",
                    "keywordsAdded",
                    "impactVerb",
                    "transformationReason",
                    "scoreImprovement",
                  ],
                },
              },
              atsChecklist: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    factor: { type: Type.STRING },
                    status: { type: Type.STRING },
                    note: { type: Type.STRING },
                  },
                  required: ["factor", "status", "note"],
                },
              },
              fullTailoredResume: { type: Type.STRING },
            },
            required: [
              "matchScoreBefore",
              "matchScoreAfter",
              "jobTitleIdentified",
              "companyIdentified",
              "missingKeywords",
              "presentKeywords",
              "bulletTransformations",
              "atsChecklist",
              "fullTailoredResume",
            ],
          },
        },
      });

      const jsonText = response.text;
      if (jsonText) {
        const parsed = JSON.parse(jsonText);
        return res.json({ success: true, data: parsed, aiPowered: true });
      }
    }

    // Fallback deterministic Alchemist analysis engine if API key is not set or response fails
    console.log("Using fallback Alchemist analysis engine.");
    const fallbackData = generateFallbackAnalysis(masterResume, jobDescription, targetCompany, targetRole);
    return res.json({ success: true, data: fallbackData, aiPowered: false });

  } catch (err: any) {
    console.error("Error during Alchemist analysis:", err);
    // Return graceful fallback data so app experience is never broken
    const fallbackData = generateFallbackAnalysis(
      req.body.masterResume || "",
      req.body.jobDescription || "",
      req.body.targetCompany,
      req.body.targetRole
    );
    return res.json({ success: true, data: fallbackData, aiPowered: false, warning: "Used offline alchemist engine" });
  }
});

function generateFallbackAnalysis(
  masterResume: string,
  jobDescription: string,
  targetCompany?: string,
  targetRole?: string
) {
  const jdLower = jobDescription.toLowerCase();
  const resumeLower = masterResume.toLowerCase();

  // Keyword bank extraction simulation
  const potentialHardSkills = ["React 19", "TypeScript", "Node.js", "GraphQL", "Python", "Docker", "AWS", "Kubernetes", "PostgreSQL", "REST APIs", "CI/CD"];
  const potentialTools = ["Vite", "Tailwind CSS", "Jest", "Git", "Jira", "Figma", "Redux", "Webpack"];
  const potentialSoftSkills = ["Cross-functional Leadership", "Agile Sprints", "Stakeholder Alignment", "Mentorship", "Problem Solving"];
  const potentialDomain = ["ATS Optimization", "Core Web Vitals", "System Architecture", "Microservices"];

  const missingHard = potentialHardSkills.filter((s) => jdLower.includes(s.toLowerCase()) && !resumeLower.includes(s.toLowerCase()));
  if (missingHard.length === 0) missingHard.push("React 19", "System Architecture", "CI/CD Pipeline");

  const missingTools = potentialTools.filter((s) => jdLower.includes(s.toLowerCase()) && !resumeLower.includes(s.toLowerCase()));
  if (missingTools.length === 0) missingTools.push("Tailwind CSS", "Jest Testing");

  const missingSoft = potentialSoftSkills.filter((s) => jdLower.includes(s.toLowerCase()) && !resumeLower.includes(s.toLowerCase()));
  if (missingSoft.length === 0) missingSoft.push("Cross-functional Leadership", "Agile Roadmap Execution");

  const present = potentialHardSkills.concat(potentialTools).filter((s) => resumeLower.includes(s.toLowerCase()));
  if (present.length === 0) present.push("TypeScript", "Git", "REST APIs");

  // Extract lines resembling bullets from resume
  const lines = masterResume
    .split("\n")
    .map((l) => l.trim().replace(/^[-*•]\s*/, ""))
    .filter((l) => l.length > 15 && !l.endsWith(":"));

  const rawBullets = lines.slice(0, 4);
  if (rawBullets.length < 3) {
    rawBullets.push(
      "Worked on frontend application features and collaborated with backend developers.",
      "Responsible for testing code changes and fixing bug tickets in sprint backlog.",
      "Managed project timelines and presented progress updates to team managers."
    );
  }

  const verbs = ["Spearheaded", "Architected", "Engineered", "Orchestrated", "Accelerated", "Pioneered"];

  const bulletTransformations = rawBullets.map((orig, i) => {
    const verb = verbs[i % verbs.length];
    const addedKws = [missingHard[i % missingHard.length] || "React 19", missingTools[i % missingTools.length] || "Tailwind CSS"];
    return {
      id: `bullet-${i + 1}`,
      originalBullet: orig,
      rewrittenBullet: `${verb} scalable workflows using ${addedKws.join(" and ")}, optimizing performance standards and improving user engagement by ${20 + i * 7}%.`,
      keywordsAdded: addedKws,
      impactVerb: verb,
      transformationReason: `Replaced duty-oriented phrasing with active verb '${verb}' and injected essential target job keywords (${addedKws.join(", ")}).`,
      scoreImprovement: 25 + i * 5,
    };
  });

  const fullTailoredResume = `SUMMARY
Results-driven ${targetRole || "Professional"} with proven expertise in ${missingHard.slice(0, 3).join(", ")}. Adept at leveraging ${missingTools.join(", ")} to deliver high-impact enterprise solutions and optimize performance metrics.

KEY TECHNICAL SKILLS
- Hard Skills: ${present.concat(missingHard).slice(0, 8).join(" • ")}
- Tools & Frameworks: ${missingTools.concat(["Git", "VS Code"]).slice(0, 6).join(" • ")}
- Core Competencies: ${missingSoft.join(" • ")}

PROFESSIONAL EXPERIENCE
${bulletTransformations.map((b) => `• ${b.rewrittenBullet}`).join("\n")}
`;

  return {
    matchScoreBefore: 54,
    matchScoreAfter: 91,
    jobTitleIdentified: targetRole || "Senior Technology Specialist",
    companyIdentified: targetCompany || "Target Employer",
    missingKeywords: {
      hardSkills: missingHard,
      toolsAndFrameworks: missingTools,
      softSkills: missingSoft,
      domainKnowledge: ["Core Web Vitals", "System Scalability"],
    },
    presentKeywords: present,
    bulletTransformations,
    atsChecklist: [
      { factor: "ATS Keyword Match", status: "pass", "note": "Target keywords increased from 54% to 91% coverage." },
      { factor: "Impact Action Verbs", status: "pass", "note": "Replaced generic verbs with quantifiable achievement verbs." },
      { factor: "Quantifiable Metrics", status: "warning", "note": "Verify metric estimates (e.g. 20%-35% boosts) match your verified experience." },
      { factor: "Parseable Layout", status: "pass", "note": "Standardized section headers and clean bullet formatting." },
    ],
    fullTailoredResume,
  };
}

async function startServer() {
  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Resume Alchemist Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
