import React from 'react';
import { motion } from 'framer-motion';
import { Laptop, Briefcase, Palette, Minimize2, Code } from 'lucide-react';
import clsx from 'clsx';

interface Template {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
}

interface TemplateSelectorProps {
  selectedTemplate: string;
  onSelectTemplate: (templateId: string) => void;
}

const templates: Template[] = [
  {
    id: 'modern',
    name: 'Modern',
    description: 'Clean design with smooth animations',
    icon: <Laptop className="w-8 h-8" />,
  },
  {
    id: 'classic',
    name: 'Classic',
    description: 'Professional corporate style',
    icon: <Briefcase className="w-8 h-8" />,
  },
  {
    id: 'creative',
    name: 'Creative',
    description: 'Colorful and dynamic layout',
    icon: <Palette className="w-8 h-8" />,
  },
  {
    id: 'minimal',
    name: 'Minimal',
    description: 'Ultra-simple focus on content',
    icon: <Minimize2 className="w-8 h-8" />,
  },
  {
    id: 'tech',
    name: 'Tech',
    description: 'Developer-friendly terminal theme',
    icon: <Code className="w-8 h-8" />,
  },
];

export const TemplateSelector: React.FC<TemplateSelectorProps> = ({
  selectedTemplate,
  onSelectTemplate,
}) => {
  return (
    <div className="w-full">
      <div className="mb-6 text-center">
        <h3 className="text-2xl font-bold text-white mb-2">Choose Your Template</h3>
        <p className="text-slate-400">Select a design that matches your style</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {templates.map((template, index) => (
          <motion.button
            key={template.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => onSelectTemplate(template.id)}
            className={clsx(
              'relative p-6 rounded-xl border-2 transition-all duration-300 text-left',
              'hover:scale-105 hover:shadow-xl',
              selectedTemplate === template.id
                ? 'border-blue-500 bg-blue-500/10 shadow-lg shadow-blue-500/20'
                : 'border-slate-700 bg-slate-800 hover:border-slate-600'
            )}
          >
            {/* Selection indicator */}
            {selectedTemplate === template.id && (
              <motion.div
                layoutId="selected-template"
                className="absolute top-3 right-3 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
              >
                <svg
                  className="w-4 h-4 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              </motion.div>
            )}

            <div
              className={clsx(
                'mb-4 transition-colors',
                selectedTemplate === template.id ? 'text-blue-400' : 'text-slate-400'
              )}
            >
              {template.icon}
            </div>

            <h4 className="text-xl font-semibold text-white mb-2">{template.name}</h4>
            <p className="text-sm text-slate-400">{template.description}</p>
          </motion.button>
        ))}
      </div>
    </div>
  );
};
