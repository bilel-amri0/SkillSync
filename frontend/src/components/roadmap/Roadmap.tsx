import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Download, Rocket } from 'lucide-react';
import type { LearningRoadmap } from '../../types/careerGuidance';
import { RoadmapDetails } from './RoadmapDetails';
import { RoadmapPhaseCard } from './RoadmapPhaseCard';

interface RoadmapProps {
  roadmap: LearningRoadmap;
  onDownload: () => void;
}

export const Roadmap = ({ roadmap, onDownload }: RoadmapProps) => {
  const [expandedPhase, setExpandedPhase] = useState<number | null>(0);
  const phases = roadmap?.phases || [];
  const totalHours = useMemo(() => {
    if (roadmap.total_time_estimate_hours !== undefined) {
      return roadmap.total_time_estimate_hours;
    }
    return phases.reduce((sum, phase) => sum + (phase.total_time_estimate_hours || 0), 0);
  }, [phases, roadmap.total_time_estimate_hours]);

  if (!phases.length) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6"
      >
        <div className="flex items-center gap-3 mb-4">
          <Rocket className="w-6 h-6 text-orange-600" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Learning Roadmap</h2>
        </div>
        <p className="text-gray-600 dark:text-gray-400">No roadmap phases are available for this profile.</p>
      </motion.div>
    );
  }

  const handleToggle = (index: number) => {
    setExpandedPhase(prev => (prev === index ? null : index));
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6"
    >
      <div className="flex flex-wrap items-center gap-3 mb-6 justify-between">
        <div className="flex items-center gap-3">
          <Rocket className="w-6 h-6 text-orange-600" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">🎯 ML-Optimized Learning Roadmap</h2>
        </div>
        <button
          type="button"
          onClick={onDownload}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-900 text-white text-sm font-semibold hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-white"
        >
          <Download className="w-4 h-4" />
          Download Plan
        </button>
      </div>

      <RoadmapDetails roadmap={{ ...roadmap, total_time_estimate_hours: totalHours }} />

      <div className="space-y-4">
        {phases.map((phase, index) => (
          <RoadmapPhaseCard
            key={phase.phase_name}
            phase={phase}
            index={index}
            isExpanded={expandedPhase === index}
            onToggle={() => handleToggle(index)}
            totalPlanHours={totalHours}
          />
        ))}
      </div>
    </motion.div>
  );
};
