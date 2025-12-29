import { useMemo } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  BookOpenCheck,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock3,
  Dumbbell,
  Hammer,
  Layers,
  Link as LinkIcon,
  Sparkles,
  Target,
} from 'lucide-react';
import type { RoadmapPhase, ResourceItem, SmartMilestone } from '../../types/careerGuidance';

interface RoadmapPhaseCardProps {
  phase: RoadmapPhase;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
  totalPlanHours: number;
}

const milestoneIconMap: Record<string, typeof CheckCircle2> = {
  exercise: Dumbbell,
  project: Hammer,
  review: BookOpenCheck,
};

const tierStyles: Record<string, string> = {
  Core: 'border-blue-200 dark:border-blue-900/40 bg-blue-50 dark:bg-blue-900/10',
  Applied: 'border-purple-200 dark:border-purple-900/40 bg-purple-50 dark:bg-purple-900/10',
  Proof: 'border-emerald-200 dark:border-emerald-900/40 bg-emerald-50 dark:bg-emerald-900/10',
};

const groupResourcesByTier = (resources: ResourceItem[]) => {
  return resources.reduce<Record<string, ResourceItem[]>>((acc, resource) => {
    const tier = resource.tier || 'Core';
    if (!acc[tier]) {
      acc[tier] = [];
    }
    acc[tier].push(resource);
    return acc;
  }, {});
};

export const RoadmapPhaseCard = ({
  phase,
  index,
  isExpanded,
  onToggle,
  totalPlanHours,
}: RoadmapPhaseCardProps) => {
  const fallbackResources = phase.resources?.length ? phase.resources : phase.learning_resources || [];
  const groupedResources = useMemo(() => groupResourcesByTier(fallbackResources), [fallbackResources]);
  const resourceEntries = Object.entries(groupedResources);
  const skills = phase.skills_to_learn || [];
  const milestones = phase.milestones || [];
  const totalHours = Math.max(phase.total_time_estimate_hours || 0, 0);
  const progressPercent = totalPlanHours > 0 ? Math.round((totalHours / totalPlanHours) * 100) : 0;
  const smartMilestones = phase.smart_milestones || [];

  return (
    <div className="border-2 border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      <button
        type="button"
        onClick={onToggle}
        className="w-full p-5 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex flex-col gap-1 text-left">
          <div className="flex items-center gap-3">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-gray-900 text-white text-sm font-semibold dark:bg-white dark:text-gray-900">
              {index + 1}
            </span>
            <div>
              <p className="text-sm uppercase tracking-wide text-gray-500 dark:text-gray-400">Phase {index + 1}</p>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">{phase.phase_name}</h3>
            </div>
          </div>
          {phase.success_justification && (
            <p className="text-sm text-gray-600 dark:text-gray-400">{phase.success_justification}</p>
          )}
          <div className="flex flex-wrap gap-4 text-xs text-gray-600 dark:text-gray-400 mt-2">
            <span className="inline-flex items-center gap-1">
              <Clock3 className="w-4 h-4" />
              {phase.duration_weeks} weeks
            </span>
            <span className="inline-flex items-center gap-1">
              <Target className="w-4 h-4" />
              {phase.success_probability}
            </span>
            <span className="inline-flex items-center gap-1">
              <Sparkles className="w-4 h-4" />
              {phase.effort_level}
            </span>
          </div>
          <div className="mt-3">
            <div className="flex items-center justify-between text-xs text-gray-600 dark:text-gray-400 mb-1">
              <span>Total effort</span>
              <span>{totalHours} hrs ({progressPercent}%)</span>
            </div>
            <div className="w-full h-2 rounded-full bg-gray-200 dark:bg-gray-700">
              <div
                className="h-2 rounded-full bg-gradient-to-r from-orange-500 to-purple-600"
                style={{ width: `${Math.min(progressPercent, 100)}%` }}
              />
            </div>
          </div>
        </div>
        {isExpanded ? <ChevronUp className="flex-shrink-0" /> : <ChevronDown className="flex-shrink-0" />}
      </button>

      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-gray-200 dark:border-gray-700"
          >
            <div className="p-5 space-y-6">
              <div>
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-2 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-blue-500" /> Skills to prioritize
                </p>
                <div className="flex flex-wrap gap-2">
                  {skills.length ? (
                    skills.map(skill => (
                      <span
                        key={skill}
                        className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 text-sm rounded-full"
                      >
                        {skill}
                      </span>
                    ))
                  ) : (
                    <span className="text-sm text-gray-500 dark:text-gray-400">No skills provided</span>
                  )}
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-3 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-purple-500" /> Tiered resources
                </p>
                {resourceEntries.length === 0 ? (
                  <p className="text-sm text-gray-500 dark:text-gray-400">No resources available.</p>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {resourceEntries.map(([tier, tierResources]) => {
                      const tierClass = tierStyles[tier] || 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800';
                      const tierHours = tierResources.reduce((sum, resource) => sum + (resource.estimated_time_hours || resource.time_hours || 0), 0);
                      return (
                        <div key={tier} className={`rounded-xl border p-4 ${tierClass}`}>
                          <div className="flex items-center justify-between">
                            <p className="text-sm font-semibold text-gray-900 dark:text-white">{tier} Tier</p>
                            <span className="text-xs text-gray-600 dark:text-gray-300">{tierHours} hrs</span>
                          </div>
                          <div className="mt-3 space-y-3">
                            {tierResources.map(resource => {
                              const link = resource.link || resource.url;
                              const content = (
                                <div>
                                  <p className="font-semibold text-gray-900 dark:text-white text-sm">{resource.title}</p>
                                  <p className="text-xs text-gray-600 dark:text-gray-400">
                                    {resource.provider} • {resource.duration}
                                  </p>
                                  <div className="flex flex-wrap gap-2 mt-2 text-xs">
                                    {resource.skill && (
                                      <span className="px-2 py-0.5 rounded-full bg-white/80 dark:bg-gray-900/30 text-gray-800 dark:text-gray-200">
                                        {resource.skill}
                                      </span>
                                    )}
                                    {resource.cost && (
                                      <span className="px-2 py-0.5 rounded-full bg-white/80 dark:bg-gray-900/30 text-gray-800 dark:text-gray-200">
                                        {resource.cost}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              );

                              return link ? (
                                <a
                                  key={resource.title}
                                  href={link}
                                  target="_blank"
                                  rel="noreferrer"
                                  className="block p-3 bg-white/80 dark:bg-gray-900/50 rounded-lg shadow-sm hover:shadow-md transition"
                                >
                                  <div className="flex items-start justify-between gap-3">
                                    {content}
                                    <LinkIcon className="w-4 h-4 text-blue-500" />
                                  </div>
                                </a>
                              ) : (
                                <div
                                  key={resource.title}
                                  className="p-3 bg-white/40 dark:bg-gray-900/30 rounded-lg"
                                >
                                  {content}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-2 flex items-center gap-2">
                    <Target className="w-4 h-4 text-green-500" /> SMART Milestones
                  </p>
                  {smartMilestones.length === 0 ? (
                    <p className="text-sm text-gray-500 dark:text-gray-400">No SMART milestones provided.</p>
                  ) : (
                    <div className="space-y-3">
                      {smartMilestones.map((milestone: SmartMilestone) => {
                        const Icon = milestoneIconMap[milestone.type?.toLowerCase()] || CheckCircle2;
                        return (
                          <div key={milestone.title} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-100 dark:border-gray-700">
                            <div className="flex items-center gap-3">
                              <Icon className="w-5 h-5 text-green-500" />
                              <div>
                                <p className="font-semibold text-gray-900 dark:text-white">{milestone.title}</p>
                                <p className="text-xs text-gray-600 dark:text-gray-400">{milestone.type} • Target: {milestone.target_metric}</p>
                                <p className="text-xs text-gray-500 dark:text-gray-400">Deadline: {milestone.deadline_hours} hrs</p>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
                <div>
                  <p className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-2 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-indigo-500" /> Additional Milestones
                  </p>
                  {milestones.length === 0 ? (
                    <p className="text-sm text-gray-500 dark:text-gray-400">No additional milestones provided.</p>
                  ) : (
                    <ul className="space-y-2">
                      {milestones.map(milestone => (
                        <li key={milestone} className="flex items-start gap-2 text-sm text-gray-600 dark:text-gray-300">
                          <CheckCircle2 className="w-4 h-4 text-indigo-500 flex-shrink-0 mt-0.5" />
                          <span>{milestone}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
