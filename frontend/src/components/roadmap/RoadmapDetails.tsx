import { Clock3, Gauge, ShieldCheck, Sparkles } from 'lucide-react';
import type { LearningRoadmap } from '../../types/careerGuidance';

interface RoadmapDetailsProps {
  roadmap: LearningRoadmap;
}

export const RoadmapDetails = ({ roadmap }: RoadmapDetailsProps) => {
  const cards = [
    {
      label: 'Total Duration',
      value: `${roadmap.total_duration_weeks} weeks`,
      hint: `≈ ${roadmap.total_duration_months} months`,
      Icon: Clock3,
    },
    {
      label: 'Total Effort',
      value: `${roadmap.total_time_estimate_hours ?? 0} hrs`,
      hint: 'Aggregated focus hours',
      Icon: Gauge,
    },
    {
      label: 'Success Probability',
      value: roadmap.predicted_success_rate,
      hint: 'Confidence from ML models',
      Icon: ShieldCheck,
    },
    {
      label: 'Personalization',
      value: roadmap.personalization_score,
      hint: roadmap.learning_strategy,
      Icon: Sparkles,
    },
  ];

  return (
    <div className="bg-gradient-to-r from-orange-50 to-purple-50 dark:from-orange-900/20 dark:to-purple-900/20 rounded-lg p-5 mb-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map(({ label, value, hint, Icon }) => (
          <div key={label} className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-full bg-white/80 dark:bg-gray-900/40 flex items-center justify-center shadow-sm">
              <Icon className="w-5 h-5 text-orange-600" />
            </div>
            <div>
              <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">{label}</p>
              <p className="text-xl font-bold text-gray-900 dark:text-white">{value}</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">{hint}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
