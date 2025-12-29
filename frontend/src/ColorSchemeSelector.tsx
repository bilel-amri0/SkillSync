import React from 'react';
import { motion } from 'framer-motion';
import clsx from 'clsx';

interface ColorScheme {
  id: string;
  name: string;
  colors: {
    primary: string;
    secondary: string;
  };
}

interface ColorSchemeSelectorProps {
  selectedScheme: string;
  onSelectScheme: (schemeId: string) => void;
}

const colorSchemes: ColorScheme[] = [
  {
    id: 'blue',
    name: 'Ocean Blue',
    colors: { primary: '#2563eb', secondary: '#0ea5e9' },
  },
  {
    id: 'green',
    name: 'Forest Green',
    colors: { primary: '#059669', secondary: '#10b981' },
  },
  {
    id: 'purple',
    name: 'Royal Purple',
    colors: { primary: '#7c3aed', secondary: '#8b5cf6' },
  },
  {
    id: 'red',
    name: 'Crimson Red',
    colors: { primary: '#dc2626', secondary: '#ef4444' },
  },
  {
    id: 'orange',
    name: 'Sunset Orange',
    colors: { primary: '#ea580c', secondary: '#f97316' },
  },
];

export const ColorSchemeSelector: React.FC<ColorSchemeSelectorProps> = ({
  selectedScheme,
  onSelectScheme,
}) => {
  return (
    <div className="w-full">
      <div className="mb-4 text-center">
        <h3 className="text-xl font-bold text-white mb-1">Choose Color Scheme</h3>
        <p className="text-sm text-slate-400">Pick your favorite color palette</p>
      </div>

      <div className="flex flex-wrap justify-center gap-4">
        {colorSchemes.map((scheme) => (
          <motion.button
            key={scheme.id}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onSelectScheme(scheme.id)}
            className={clsx(
              'relative flex flex-col items-center gap-2 p-4 rounded-xl transition-all',
              selectedScheme === scheme.id
                ? 'bg-slate-700 shadow-lg'
                : 'bg-slate-800 hover:bg-slate-700'
            )}
          >
            <div className="flex gap-2">
              <div
                className="w-10 h-10 rounded-lg shadow-lg"
                style={{ backgroundColor: scheme.colors.primary }}
              />
              <div
                className="w-10 h-10 rounded-lg shadow-lg"
                style={{ backgroundColor: scheme.colors.secondary }}
              />
            </div>
            <span
              className={clsx(
                'text-sm font-medium',
                selectedScheme === scheme.id ? 'text-white' : 'text-slate-400'
              )}
            >
              {scheme.name}
            </span>

            {selectedScheme === scheme.id && (
              <motion.div
                layoutId="selected-color"
                className="absolute -top-1 -right-1 w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
              >
                <svg
                  className="w-3 h-3 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={3}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              </motion.div>
            )}
          </motion.button>
        ))}
      </div>
    </div>
  );
};
