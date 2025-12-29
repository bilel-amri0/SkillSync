import { useEffect, useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  Upload,
  Sparkles,
  Download,
  Mail,
  Phone,
  MapPin,
  Github,
  Linkedin,
  Layers,
  Code,
  Briefcase,
  Target,
  Shield,
  ArrowRight,
  Palette,
  LayoutGrid,
  FileCode2,
  ListChecks
} from 'lucide-react';
import toast from 'react-hot-toast';
import { cvApi, handleApiError } from '../services/api';
import type { CVAnalysis, PortfolioMeta, PortfolioResponsePayload } from '../types';

interface Experience {
  role: string;
  company: string;
  period: string;
  bullets: string[];
}

interface Project {
  name: string;
  description: string;
  tech: string[];
}

interface PortfolioData {
  name: string;
  title: string;
  summary: string;
  location: string;
  email: string;
  phone: string;
  linkedin?: string;
  github?: string;
  resumeUrl?: string;
  skills: Record<string, string[]>;
  experiences: Experience[];
  projects: Project[];
  certifications: string[];
}

const defaultPortfolio: PortfolioData = {
  name: 'John Doe',
  title: 'Software Engineer',
  summary:
    'Experienced Full Stack Developer with 5+ years of expertise in building scalable web applications.',
  location: 'New York, NY',
  email: 'john.doe@email.com',
  phone: '+1-555-0123',
  linkedin: undefined,
  github: undefined,
  resumeUrl: '/test_cv_sample.txt',
  skills: {
    Frontend: ['JavaScript', 'TypeScript', 'React'],
    Backend: ['Node.js', 'Express', 'Python'],
    Databases: ['PostgreSQL', 'MongoDB'],
    DevOps: ['AWS', 'Docker', 'CI/CD'],
    Quality: ['Jest', 'Cypress'],
    Collaboration: ['Git', 'Agile/Scrum']
  },
  experiences: [
    {
      role: 'Senior Software Engineer',
      company: 'TechCorp Inc.',
      period: '2020 – Present',
      bullets: [
        'Developed and maintained multiple React applications serving 100K+ users.',
        'Implemented RESTful APIs using Node.js and Express.',
        'Led a team of 4 developers in an agile environment.',
        'Reduced application load time by 40% through optimization.'
      ]
    },
    {
      role: 'Full Stack Developer',
      company: 'StartupXYZ',
      period: '2018 – 2020',
      bullets: [
        'Built an e-commerce platform from scratch using the MERN stack.',
        'Integrated payment gateways and third-party APIs.',
        'Implemented automated testing with Jest and Cypress.',
        'Collaborated with UX designers to improve user experience.'
      ]
    }
  ],
  projects: [
    {
      name: 'Scalable React Platforms',
      description:
        'Led the delivery of multiple React applications serving 100K+ users while orchestrating backend integrations and performance optimization.',
      tech: ['React', 'TypeScript', 'Node.js', 'Express']
    },
    {
      name: 'MERN Commerce Suite',
      description:
        'Built an end-to-end commerce platform with secure payments, third-party APIs, and automated testing coverage for StartupXYZ.',
      tech: ['MongoDB', 'Express', 'React', 'Node.js', 'Cypress']
    }
  ],
  certifications: ['AWS Certified Solutions Architect', 'MongoDB Certified Developer']
};

const templateOptions = [
  {
    id: 'modern',
    label: 'Nocturne Neon',
    description: 'High-contrast hero with neon gradients for bold storytelling.',
    fontFamily: 'Space Grotesk, Inter, sans-serif',
    darkMode: true,
  },
  {
    id: 'minimal',
    label: 'Clarity Minimal',
    description: 'Editorial white layout with calm typography and soft dividers.',
    fontFamily: 'Sora, Inter, sans-serif',
    darkMode: false,
  },
  {
    id: 'tech',
    label: 'Helix Techstack',
    description: 'Layered cards with grid accents for product-minded profiles.',
    fontFamily: 'IBM Plex Sans, Inter, sans-serif',
    darkMode: false,
  },
];

const colorThemes = [
  { id: 'blue', label: 'Electric Blue', gradient: 'from-sky-500 via-blue-600 to-cyan-400' },
  { id: 'teal', label: 'Aurora Teal', gradient: 'from-emerald-400 via-teal-500 to-cyan-300' },
  { id: 'rose', label: 'Infrared Rose', gradient: 'from-rose-500 via-pink-500 to-orange-400' },
  { id: 'slate', label: 'Graphite Slate', gradient: 'from-slate-700 via-slate-900 to-gray-800' },
];

const SECTION_OPTIONS = [
  { id: 'about', label: 'About' },
  { id: 'skills', label: 'Skills' },
  { id: 'experience', label: 'Experience' },
  { id: 'projects', label: 'Projects' },
  { id: 'education', label: 'Education' },
  { id: 'impact', label: 'Impact' },
  { id: 'roadmap', label: 'Roadmap' },
  { id: 'opportunities', label: 'Opportunities' },
  { id: 'contact', label: 'Contact' },
];

const extractSection = (source: string, startToken: string, endToken?: string) => {
  const upper = source.toUpperCase();
  const startIndex = upper.indexOf(startToken.toUpperCase());
  if (startIndex === -1) return '';
  const afterStart = source.slice(startIndex + startToken.length);
  if (!endToken) return afterStart.trim();
  const endIndex = afterStart.toUpperCase().indexOf(endToken.toUpperCase());
  return (endIndex === -1 ? afterStart : afterStart.slice(0, endIndex)).trim();
};

const splitBullets = (section: string) =>
  section
    .split(/\r?\n/)
    .map(line => line.replace(/^[-•]\s*/, '').trim())
    .filter(Boolean);

const categorizeSkills = (skills: string[]) => {
  const groups: Record<string, string[]> = {
    Frontend: [],
    Backend: [],
    Databases: [],
    DevOps: [],
    Quality: [],
    Collaboration: [],
    Other: []
  };

  const mapping: Record<string, keyof typeof groups> = {
    javascript: 'Frontend',
    typescript: 'Frontend',
    react: 'Frontend',
    'node.js': 'Backend',
    express: 'Backend',
    python: 'Backend',
    postgresql: 'Databases',
    mongodb: 'Databases',
    aws: 'DevOps',
    docker: 'DevOps',
    'ci/cd': 'DevOps',
    jest: 'Quality',
    cypress: 'Quality',
    git: 'Collaboration',
    'agile/scrum': 'Collaboration'
  };

  skills.forEach(raw => {
    raw
      .split(',')
      .map(skill => skill.trim())
      .filter(Boolean)
      .forEach(skill => {
        const key = skill.toLowerCase();
        const target = Object.keys(mapping).find(name => key.includes(name));
        const bucket = target ? mapping[target] : 'Other';
        if (!groups[bucket].includes(skill)) {
          groups[bucket].push(skill);
        }
      });
  });

  return Object.fromEntries(
    Object.entries(groups).filter(([, values]) => values.length > 0)
  );
};

const parseExperiences = (section: string, fallback: Experience[]) => {
  const lines = section.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  const experiences: Experience[] = [];
  let current: Experience | null = null;

  lines.forEach(line => {
    if (line.includes('|') && !line.startsWith('-')) {
      if (current) {
        experiences.push(current);
      }
      const [role = '', company = '', period = ''] = line.split('|').map(part => part.trim());
      current = {
        role,
        company,
        period: period || '',
        bullets: []
      };
    } else if (line.startsWith('-') && current) {
      current.bullets.push(line.replace(/^-\s*/, '').trim());
    }
  });

  if (current) {
    experiences.push(current);
  }

  return experiences.length ? experiences : fallback;
};

const buildProjects = (experiences: Experience[], skills: Record<string, string[]>, fallback: Project[]) => {
  if (!experiences.length) {
    return fallback;
  }

  const techPool = Array.from(
    new Set(
      Object.values(skills)
        .flat()
        .slice(0, 6)
    )
  );

  const derived = experiences.map(exp => ({
    name: `${exp.role} @ ${exp.company}`,
    description: exp.bullets.slice(0, 2).join(' ') || exp.role,
    tech: techPool.length ? techPool : fallback[0]?.tech || []
  }));

  return derived.length ? derived : fallback;
};

const parsePortfolioFromCv = (text: string): PortfolioData => {
  const lines = text.split(/\r?\n/).map(line => line.trim()).filter(Boolean);

  const name = lines[0] || defaultPortfolio.name;
  const title = lines[1] || defaultPortfolio.title;
  const contactLine = lines.find(line => line.includes('@')) || '';
  const [email = defaultPortfolio.email, phone = defaultPortfolio.phone, location = defaultPortfolio.location] =
    contactLine.split('|').map(part => part.trim());

  const summary =
    extractSection(text, 'PROFESSIONAL SUMMARY', 'SKILLS') || defaultPortfolio.summary;

  const skillsSection = extractSection(text, 'SKILLS', 'WORK EXPERIENCE');
  const skillLines = splitBullets(skillsSection);
  const skills = skillLines.length
    ? categorizeSkills(skillLines)
    : defaultPortfolio.skills;

  const experiencesSection = extractSection(text, 'WORK EXPERIENCE', 'EDUCATION');
  const experiences = parseExperiences(experiencesSection, defaultPortfolio.experiences);

  const projects = buildProjects(experiences, skills, defaultPortfolio.projects);

  const certSection = extractSection(text, 'CERTIFICATIONS');
  const certifications = splitBullets(certSection).length
    ? splitBullets(certSection)
    : defaultPortfolio.certifications;

  const linkedinLine = lines.find(line => /linkedin/i.test(line));
  const githubLine = lines.find(line => /github/i.test(line));

  const linkedin = linkedinLine ? linkedinLine.split(/\s+/).find(token => token.includes('http')) : undefined;
  const github = githubLine ? githubLine.split(/\s+/).find(token => token.includes('http')) : undefined;

  return {
    name,
    title,
    summary,
    location,
    email,
    phone,
    linkedin,
    github,
    resumeUrl: defaultPortfolio.resumeUrl,
    skills,
    experiences,
    projects,
    certifications
  };
};

const PortfolioPreview = ({
  data,
  downloadUrl,
  templateId,
  accentTheme,
  htmlDownloadUrl,
  backendMeta,
}: {
  data: PortfolioData;
  downloadUrl?: string;
  templateId: string;
  accentTheme: string;
  htmlDownloadUrl?: string | null;
  backendMeta?: PortfolioMeta | null;
}) => {
  const resumeHref = downloadUrl || data.resumeUrl || '/test_cv_sample.txt';
  const templateMeta = useMemo(() => templateOptions.find(option => option.id === templateId) ?? templateOptions[0], [templateId]);
  const gradient = useMemo(() => colorThemes.find(theme => theme.id === accentTheme)?.gradient ?? colorThemes[0].gradient, [accentTheme]);

  if (backendMeta) {
    const hero = backendMeta.hero ?? {};
    const skills = backendMeta.skills ?? {};
    const experiences = backendMeta.experiences ?? [];
    const projects = backendMeta.projects ?? [];
    const education = backendMeta.education ?? [];
    const contact = hero.contact ?? {};
    const stats = backendMeta.stats;
    const heroName = hero.name || backendMeta.name || data.name;
    const heroTitle = hero.title || backendMeta.title || data.title;
    const heroLocation = hero.location || backendMeta.location || data.location;
    const heroSummary = hero.headline || backendMeta.summary || data.summary;

    const statItems = [
      stats?.experience_years ? { label: 'Experience', value: `${stats.experience_years.toFixed(1)} yrs` } : null,
      stats?.skills_count ? { label: 'Skills', value: stats.skills_count } : null,
      stats?.sections ? { label: 'Sections', value: stats.sections.length } : null,
    ].filter(Boolean) as Array<{ label: string; value: string | number }>;

    const contactChips = Object.entries(contact)
      .filter(([, value]) => Boolean(value))
      .map(([label, value]) => (
        <span key={label} className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm text-white/90">
          <span className="text-xs uppercase tracking-[0.2em] text-white/60">{label}</span>
          <span>{value}</span>
        </span>
      ));

    return (
      <div className="space-y-6">
        <section className="relative overflow-hidden rounded-3xl bg-slate-950 p-8 text-white">
          <div className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-40 blur-3xl`} />
          <div className="relative space-y-4">
            <p className="text-xs uppercase tracking-[0.4em] text-white/70">Featured Portfolio</p>
            <h1 className="text-4xl font-semibold">{heroName}</h1>
            <p className="text-lg text-white/80">{heroTitle} · {heroLocation}</p>
            <p className="text-white/90 max-w-3xl">{heroSummary}</p>
            <div className="flex flex-wrap gap-2">{contactChips}</div>
            {statItems.length ? (
              <div className="flex flex-wrap gap-4 pt-2">
                {statItems.map(item => (
                  <div key={item.label} className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                    <p className="text-2xl font-semibold text-white">{item.value}</p>
                    <p className="text-xs uppercase tracking-[0.3em] text-white/60">{item.label}</p>
                  </div>
                ))}
              </div>
            ) : null}
            <div className="flex flex-wrap gap-3 pt-4">
              <a href={resumeHref} download className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-3 text-slate-900">
                <Download className="h-5 w-5" />
                Download CV
              </a>
              <a href={`mailto:${contact.email || data.email}`} className="inline-flex items-center gap-2 rounded-full border border-white/30 px-5 py-3">
                <Sparkles className="h-5 w-5" />
                Book a call
              </a>
              {htmlDownloadUrl && (
                <a
                  href={htmlDownloadUrl}
                  download={`portfolio-${(heroName || 'portfolio').toLowerCase().replace(/\s+/g, '-')}.html`}
                  className="inline-flex items-center gap-2 rounded-full border border-emerald-200/60 bg-emerald-400/20 px-5 py-3 text-emerald-100"
                >
                  <FileCode2 className="h-5 w-5" />
                  Download Web Template
                </a>
              )}
            </div>
          </div>
        </section>

        {backendMeta.summary ? (
          <section className="rounded-3xl bg-white p-8 text-slate-900 shadow-xl">
            <div className="flex items-center gap-3 mb-4">
              <Layers className="h-6 w-6 text-blue-600" />
              <h2 className="text-2xl font-semibold">About</h2>
            </div>
            <p className="text-slate-600 leading-relaxed">{backendMeta.summary}</p>
          </section>
        ) : null}

        {Object.keys(skills).length ? (
          <section className="rounded-3xl bg-white p-8 text-slate-900 shadow-xl">
            <div className="flex items-center gap-3 mb-4">
              <Code className="h-6 w-6 text-indigo-600" />
              <h2 className="text-2xl font-semibold">Skills</h2>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              {Object.entries(skills).map(([category, values]) => (
                <div key={category} className="rounded-2xl border border-slate-200 p-4">
                  <p className="text-sm uppercase tracking-[0.3em] text-slate-400">{category}</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {values.map(value => (
                      <span key={`${category}-${value}`} className="rounded-full bg-slate-100 px-3 py-1 text-sm text-slate-700">
                        {value}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        ) : null}

        {experiences.length ? (
          <section className="rounded-3xl bg-white p-8 text-slate-900 shadow-xl">
            <div className="flex items-center gap-3 mb-4">
              <Briefcase className="h-6 w-6 text-purple-600" />
              <h2 className="text-2xl font-semibold">Experience</h2>
            </div>
            <div className="space-y-6">
              {experiences.map(exp => (
                <div key={`${exp.company}-${exp.title}`} className="rounded-2xl border border-slate-100 p-5">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-xs uppercase tracking-[0.3em] text-slate-400">{exp.company}</p>
                      <h3 className="text-xl font-semibold">{exp.title}</h3>
                    </div>
                    <p className="text-sm text-slate-500">{exp.period}</p>
                  </div>
                  <ul className="mt-4 space-y-2 text-sm text-slate-600">
                    {exp.bullets.map((bullet, idx) => (
                      <li key={`${exp.title}-${idx}`} className="flex gap-2">
                        <ArrowRight className="h-4 w-4 text-purple-500" />
                        <span>{bullet}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        ) : null}

        {projects.length ? (
          <section className="rounded-3xl bg-white p-8 text-slate-900 shadow-xl">
            <div className="flex items-center gap-3 mb-4">
              <Target className="h-6 w-6 text-cyan-600" />
              <h2 className="text-2xl font-semibold">Projects</h2>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              {projects.map(project => (
                <div key={project.title} className="rounded-2xl border border-slate-100 p-5">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">{project.title}</h3>
                    <Shield className="h-5 w-5 text-cyan-500" />
                  </div>
                  {project.summary && <p className="mt-2 text-sm text-slate-600">{project.summary}</p>}
                  {project.bullets?.length ? (
                    <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-600">
                      {project.bullets.map((bullet, idx) => (
                        <li key={`${project.title}-${idx}`}>{bullet}</li>
                      ))}
                    </ul>
                  ) : null}
                  {project.tech?.length ? (
                    <div className="mt-4 flex flex-wrap gap-2">
                      {project.tech.map(tag => (
                        <span key={`${project.title}-${tag}`} className="rounded-full bg-cyan-50 px-3 py-1 text-xs font-medium text-cyan-700">
                          {tag}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          </section>
        ) : null}

        {education.length ? (
          <section className="rounded-3xl bg-white p-8 text-slate-900 shadow-xl">
            <div className="flex items-center gap-3 mb-4">
              <Layers className="h-6 w-6 text-amber-500" />
              <h2 className="text-2xl font-semibold">Education</h2>
            </div>
            <ul className="space-y-3 text-slate-600">
              {education.map(entry => (
                <li key={entry} className="rounded-2xl border border-slate-100 p-4 text-sm">
                  {entry}
                </li>
              ))}
            </ul>
          </section>
        ) : null}
      </div>
    );
  }

  const darkMode = templateMeta.darkMode;
  const panelText = darkMode ? 'text-white' : 'text-slate-900';
  const mutedText = darkMode ? 'text-slate-300' : 'text-slate-600';
  const borderTone = darkMode ? 'border-white/10' : 'border-slate-200';
  const chipTone = darkMode ? 'bg-white/10 text-white' : 'bg-slate-100 text-slate-800';
  const iconTone = darkMode ? 'text-white/80' : 'text-slate-600';

  return (
    <div className="space-y-6">
      <section className={`relative overflow-hidden rounded-3xl ${darkMode ? 'bg-slate-950 text-white' : 'bg-white text-slate-900'} p-8`} style={{ fontFamily: templateMeta.fontFamily }}>
        <div className={`absolute inset-0 bg-gradient-to-r ${gradient} ${darkMode ? 'opacity-40' : 'opacity-30'} blur-3xl`} />
        <div className="relative flex flex-col gap-6">
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-white/70">Portfolio</p>
            <h1 className="text-4xl font-semibold mt-2">{data.name}</h1>
            <p className={`text-xl ${mutedText}`}>{data.title}</p>
          </div>
          <p className={`${mutedText} max-w-3xl`}>{data.summary}</p>
          <div className={`flex flex-wrap items-center gap-3 text-sm ${mutedText}`}>
            <span className={`inline-flex items-center gap-2 rounded-full ${chipTone} px-3 py-1`}>
              <Mail className={`h-4 w-4 ${iconTone}`} />
              {data.email}
            </span>
            <span className={`inline-flex items-center gap-2 rounded-full ${chipTone} px-3 py-1`}>
              <Phone className={`h-4 w-4 ${iconTone}`} />
              {data.phone}
            </span>
            <span className={`inline-flex items-center gap-2 rounded-full ${chipTone} px-3 py-1`}>
              <MapPin className={`h-4 w-4 ${iconTone}`} />
              {data.location}
            </span>
          </div>
          <div className="flex flex-wrap gap-4">
            <a href={resumeHref} download className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-3 text-slate-900 font-semibold shadow-lg shadow-white/20">
              <Download className="h-5 w-5" />
              Download CV
            </a>
            <a href={`mailto:${data.email}`} className={`inline-flex items-center gap-2 rounded-full border ${darkMode ? 'border-white/40 text-white' : 'border-slate-300 text-slate-900'} px-5 py-3`}>
              <Sparkles className="h-5 w-5" />
              Book a call
            </a>
            {htmlDownloadUrl && (
              <a
                href={htmlDownloadUrl}
                download={`portfolio-${data.name.toLowerCase().replace(/\s+/g, '-')}.html`}
                className={`inline-flex items-center gap-2 rounded-full border border-emerald-200/60 bg-emerald-400/10 px-5 py-3 ${darkMode ? 'text-emerald-100' : 'text-emerald-900'}`}
              >
                <FileCode2 className="h-5 w-5" />
                Download Web Template
              </a>
            )}
          </div>
        </div>
      </section>

      {/* Fallback sections */}
      <section className={`rounded-3xl ${darkMode ? 'bg-white/5' : 'bg-white'} ${panelText} p-8 shadow-xl border ${borderTone}`}>
        <div className="flex items-center gap-3 mb-6">
          <Layers className="h-6 w-6 text-blue-500" />
          <h2 className="text-2xl font-semibold">About</h2>
        </div>
        <p className={`${mutedText} leading-relaxed`}>{data.summary}</p>
      </section>

      <section className={`rounded-3xl ${darkMode ? 'bg-white/5' : 'bg-white'} ${panelText} p-8 shadow-xl border ${borderTone}`}>
        <div className="flex items-center gap-3 mb-6">
          <Code className="h-6 w-6 text-indigo-500" />
          <h2 className="text-2xl font-semibold">Tech Stack</h2>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {Object.entries(data.skills).map(([category, values]) => (
            <div key={category} className={`rounded-2xl border ${borderTone} p-4`}>
              <p className={`text-sm uppercase tracking-widest ${mutedText}`}>{category}</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {values.map(skill => (
                  <span key={`${category}-${skill}`} className={`rounded-full px-3 py-1 text-sm ${chipTone}`}>
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className={`rounded-3xl ${darkMode ? 'bg-white/5' : 'bg-white'} ${panelText} p-8 shadow-xl border ${borderTone}`}>
        <div className="flex items-center gap-3 mb-6">
          <Briefcase className="h-6 w-6 text-purple-500" />
          <h2 className="text-2xl font-semibold">Experience</h2>
        </div>
        <div className={`relative ml-4 border-l ${borderTone}`}>
          {data.experiences.map((exp, idx) => (
            <div key={`${exp.company}-${exp.role}`} className="relative pl-8 pb-8">
              <div className="absolute -left-3 flex h-6 w-6 items-center justify-center rounded-full bg-purple-600 text-white">
                {idx + 1}
              </div>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className={`text-sm uppercase tracking-widest ${mutedText}`}>{exp.company}</p>
                  <h3 className="text-xl font-semibold">{exp.role}</h3>
                </div>
                <p className={`text-sm ${mutedText}`}>{exp.period}</p>
              </div>
              <ul className={`mt-4 space-y-2 text-sm ${mutedText}`}>
                {exp.bullets.map((bullet, bulletIdx) => (
                  <li key={`${exp.company}-${bulletIdx}`} className="flex gap-2">
                    <ArrowRight className="h-4 w-4 text-purple-500" />
                    <span>{bullet}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section className={`rounded-3xl ${darkMode ? 'bg-white/5' : 'bg-white'} ${panelText} p-8 shadow-xl border ${borderTone}`}>
        <div className="flex items-center gap-3 mb-6">
          <Target className="h-6 w-6 text-cyan-500" />
          <h2 className="text-2xl font-semibold">Projects</h2>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {data.projects.map(project => (
            <div key={project.name} className={`rounded-2xl border ${borderTone} p-5`}>
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">{project.name}</h3>
                <Shield className="h-5 w-5 text-cyan-500" />
              </div>
              <p className={`mt-2 text-sm ${mutedText}`}>{project.description}</p>
              <div className="mt-4 flex flex-wrap gap-2">
                {project.tech.map(tech => (
                  <span key={`${project.name}-${tech}`} className={`rounded-full px-3 py-1 text-xs font-medium ${chipTone}`}>
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className={`rounded-3xl ${darkMode ? 'bg-white/5' : 'bg-white'} ${panelText} p-8 shadow-xl border ${borderTone}`}>
        <h2 className="text-2xl font-semibold mb-4">Certifications</h2>
        <div className="flex flex-wrap gap-3">
          {data.certifications.map(cert => (
            <span key={cert} className={`rounded-full border ${borderTone} px-4 py-2 text-sm ${mutedText}`}>
              {cert}
            </span>
          ))}
        </div>
      </section>
    </div>
  );
};

export const CVAnalysisPage = () => {
  const queryClient = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [portfolioData, setPortfolioData] = useState<PortfolioData>(defaultPortfolio);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string>(defaultPortfolio.resumeUrl || '');
  const [selectedTemplate, setSelectedTemplate] = useState<string>(templateOptions[0].id);
  const [selectedTheme, setSelectedTheme] = useState<string>(colorThemes[0].id);
  const [selectedSections, setSelectedSections] = useState<string[]>(SECTION_OPTIONS.map(option => option.id));
  const [backendPortfolio, setBackendPortfolio] = useState<PortfolioResponsePayload | null>(null);
  const [htmlDownloadUrl, setHtmlDownloadUrl] = useState<string | null>(null);
  const [analysisPayload, setAnalysisPayload] = useState<CVAnalysis | null>(null);

  const runAdvancedAnalysis = async (cvText: string) => {
    try {
      const analysis = await cvApi.analyzeAdvanced(cvText);
      setAnalysisPayload(analysis);
      toast.success('Advanced ML analysis synced');
      return analysis;
    } catch (apiError) {
      const message = handleApiError(apiError) || 'Failed to sync CV data';
      setError(message);
      toast.error(message);
      throw apiError;
    }
  };

  const persistCvFile = async (cvFile: File) => {
    const extension = cvFile.name.split('.').pop()?.toLowerCase();
    if (!extension || !['pdf', 'docx'].includes(extension)) {
      return null;
    }
    try {
      const response = await cvApi.uploadCv(cvFile);
      toast.success('CV stored server-side for analytics');
      return response.analysis_id;
    } catch (uploadError) {
      const message = handleApiError(uploadError) || 'Failed to persist CV analysis';
      toast.error(message);
      return null;
    }
  };

  const updateHtmlDownloadUrl = (htmlContent: string) => {
    setHtmlDownloadUrl(prev => {
      if (prev) {
        URL.revokeObjectURL(prev);
      }
      return URL.createObjectURL(new Blob([htmlContent], { type: 'text/html' }));
    });
  };

  useEffect(() => {
    if (!file) {
      setDownloadUrl(defaultPortfolio.resumeUrl || '');
      return;
    }
    const url = URL.createObjectURL(file);
    setDownloadUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    return () => {
      if (htmlDownloadUrl) {
        URL.revokeObjectURL(htmlDownloadUrl);
      }
    };
  }, [htmlDownloadUrl]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setFile(event.target.files[0]);
      setError(null);
      setBackendPortfolio(null);
      setAnalysisPayload(null);
      setHtmlDownloadUrl(prev => {
        if (prev) {
          URL.revokeObjectURL(prev);
        }
        return null;
      });
    }
  };

  const handleSectionToggle = (sectionId: string) => {
    setSelectedSections(prev => {
      if (prev.includes(sectionId)) {
        if (prev.length === 1) {
          toast.error('Keep at least one section visible');
          return prev;
        }
        return prev.filter(id => id !== sectionId);
      }
      return [...prev, sectionId];
    });
  };

  const handleGeneratePortfolio = async () => {
    if (!file) {
      setError('Please select a CV file (.txt, .pdf, or .docx).');
      return;
    }
    if (!selectedSections.length) {
      setError('Select at least one portfolio section.');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setBackendPortfolio(null);

    try {
      let extractedText = '';
      try {
        const { cv_text } = await cvApi.extractText(file);
        extractedText = cv_text;
      } catch (extractionError) {
        console.warn('extract-text fallback to File.text()', extractionError);
        extractedText = await file.text();
      }

      if (!extractedText.trim()) {
        throw new Error('Could not extract any text from the uploaded CV');
      }

      const parsed = parsePortfolioFromCv(extractedText);
      setPortfolioData(parsed);

      const analysis = await runAdvancedAnalysis(extractedText);
      const templateMeta = templateOptions.find(option => option.id === selectedTemplate) ?? templateOptions[0];
      const customizationPayload = {
        color_scheme: selectedTheme,
        font_family: templateMeta.fontFamily.split(',')[0] || 'Inter',
        layout_style: selectedTemplate,
        sections_visible: selectedSections,
        include_photo: true,
        include_projects: selectedSections.includes('projects'),
        include_contact_form: selectedSections.includes('contact'),
        dark_mode: templateMeta.darkMode,
      };

      await persistCvFile(file);

      const portfolio = await cvApi.generatePortfolio({
        cvId: analysis.analysis_id,
        templateId: selectedTemplate,
        customization: customizationPayload,
      });
      setBackendPortfolio(portfolio);
      updateHtmlDownloadUrl(portfolio.html_content);

      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['cv-analyses'] }),
        queryClient.invalidateQueries({ queryKey: ['dashboard-latest'] }),
      ]);
      toast.success('Portfolio generated. Ready for preview.');
    } catch (err) {
      console.error(err);
      const message = handleApiError(err) || 'Failed to process the CV. Please upload a plain-text resume.';
      setError(message);
      toast.error(message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleUseSample = () => {
    setFile(null);
    setPortfolioData(defaultPortfolio);
    setError(null);
    setBackendPortfolio(null);
    setAnalysisPayload(null);
    setHtmlDownloadUrl(prev => {
      if (prev) {
        URL.revokeObjectURL(prev);
      }
      return null;
    });
  };

  const highlightStats = useMemo(() => {
    const stats: Array<{ label: string; value: string | number }> = [
      { label: 'Experience Entries', value: portfolioData.experiences.length },
      { label: 'Projects', value: portfolioData.projects.length },
      { label: 'Core Skills', value: Object.values(portfolioData.skills).flat().length },
      { label: 'Certifications', value: portfolioData.certifications.length },
    ];
    if (backendPortfolio?.portfolio?.stats?.skills_count) {
      stats.push({ label: 'Backend Skills', value: backendPortfolio.portfolio.stats.skills_count });
    }
    if (analysisPayload?.confidence_score) {
      stats.push({ label: 'Confidence', value: `${Math.round(analysisPayload.confidence_score * 100)}%` });
    }
    return stats;
  }, [portfolioData, backendPortfolio, analysisPayload]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto grid gap-6 lg:grid-cols-[380px,1fr]">
        <div className="space-y-6">
          <section className="rounded-3xl bg-white/10 backdrop-blur p-6">
            <div className="flex items-center gap-3 mb-4">
              <Sparkles className="h-6 w-6 text-blue-200" />
              <h2 className="text-xl font-semibold">Generate Portfolio from CV</h2>
            </div>
            <div className="rounded-2xl border-2 border-dashed border-white/30 p-6 text-center">
              <Upload className="mx-auto h-12 w-12 text-white/50" />
              <p className="mt-3 text-sm text-white/70">Upload a TXT, PDF, or DOCX resume for full-text extraction.</p>
              <input
                type="file"
                accept=".txt,.pdf,.doc,.docx"
                id="cv-upload"
                className="hidden"
                onChange={handleFileChange}
              />
              <label
                htmlFor="cv-upload"
                className="mt-4 inline-flex items-center justify-center rounded-full bg-white px-5 py-3 text-slate-900 font-semibold cursor-pointer"
              >
                Choose File
              </label>
              {file && <p className="mt-3 text-sm text-white/80">{file.name}</p>}
              {error && <p className="mt-3 text-sm text-red-300">{error}</p>}
              <button
                type="button"
                onClick={handleGeneratePortfolio}
                disabled={isProcessing}
                className="mt-6 w-full rounded-full bg-blue-500 py-3 font-semibold text-white disabled:opacity-50"
              >
                {isProcessing ? 'Processing CV...' : 'Generate Portfolio'}
              </button>
              <button
                type="button"
                onClick={handleUseSample}
                className="mt-3 w-full rounded-full border border-white/30 py-3 font-semibold text-white"
              >
                Use Sample CV Data
              </button>
            </div>
          </section>

          <section className="rounded-3xl bg-white/10 backdrop-blur p-6">
            <div className="flex items-center gap-3 mb-4">
              <LayoutGrid className="h-6 w-6 text-purple-200" />
              <h3 className="text-lg font-semibold">Template & Theme</h3>
            </div>
            <div className="space-y-3">
              {templateOptions.map(option => {
                const isActive = option.id === selectedTemplate;
                return (
                  <button
                    key={option.id}
                    type="button"
                    onClick={() => setSelectedTemplate(option.id)}
                    className={`w-full rounded-2xl border px-4 py-3 text-left transition hover:border-white/50 ${
                      isActive ? 'border-white bg-white/10' : 'border-white/10 bg-white/5'
                    }`}
                  >
                    <p className="font-semibold text-white">{option.label}</p>
                    <p className="text-xs text-white/70">{option.description}</p>
                  </button>
                );
              })}
            </div>

            <div className="mt-6">
              <p className="text-sm text-white/70 mb-2 inline-flex items-center gap-2">
                <Palette className="h-4 w-4" /> Accent palette
              </p>
              <div className="flex flex-wrap gap-2">
                {colorThemes.map(theme => {
                  const isActive = selectedTheme === theme.id;
                  return (
                    <button
                      key={theme.id}
                      type="button"
                      onClick={() => setSelectedTheme(theme.id)}
                      className={`h-10 w-16 rounded-xl border ${isActive ? 'border-white' : 'border-transparent'} bg-gradient-to-r ${theme.gradient}`}
                    />
                  );
                })}
              </div>
            </div>

            <div className="mt-6">
              <p className="text-sm text-white/70 mb-2 inline-flex items-center gap-2">
                <ListChecks className="h-4 w-4" /> Portfolio sections
              </p>
              <div className="grid grid-cols-2 gap-2">
                {SECTION_OPTIONS.map(section => {
                  const isActive = selectedSections.includes(section.id);
                  return (
                    <button
                      key={section.id}
                      type="button"
                      onClick={() => handleSectionToggle(section.id)}
                      className={`rounded-xl border px-3 py-2 text-sm font-medium transition ${
                        isActive ? 'border-emerald-300 bg-emerald-400/20 text-white' : 'border-white/10 text-white/70'
                      }`}
                    >
                      {section.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </section>

          <section className="rounded-3xl bg-white/10 backdrop-blur p-6">
            <div className="flex items-center gap-3 mb-4">
              <Shield className="h-6 w-6 text-teal-200" />
              <h3 className="text-lg font-semibold">Highlights</h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {highlightStats.map(stat => (
                <div key={stat.label} className="rounded-2xl bg-white/5 p-3 text-center">
                  <p className="text-3xl font-bold text-white">{stat.value}</p>
                  <p className="text-xs uppercase tracking-wide text-white/60">{stat.label}</p>
                </div>
              ))}
            </div>
          </section>
        </div>

        <PortfolioPreview
          data={portfolioData}
          downloadUrl={downloadUrl}
          templateId={selectedTemplate}
          accentTheme={selectedTheme}
          backendMeta={backendPortfolio?.portfolio}
          htmlDownloadUrl={htmlDownloadUrl}
        />
      </div>
    </div>
  );
};
