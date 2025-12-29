import React, { useCallback, useState } from 'react';
import { Upload, FileText, X, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';

interface CVUploaderProps {
  onFileSelect: (file: File) => void;
  isAnalyzing: boolean;
}

export const CVUploader: React.FC<CVUploaderProps> = ({ onFileSelect, isAnalyzing }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const validateFile = (file: File): boolean => {
    const validTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/msword',
      'text/plain',
    ];

    const validExtensions = ['.pdf', '.docx', '.doc', '.txt'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
      setError('Please upload a PDF, DOCX, or TXT file');
      return false;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return false;
    }

    setError(null);
    return true;
  };

  const handleFile = useCallback((file: File) => {
    if (validateFile(file)) {
      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFile(files[0]);
    }
  }, [handleFile]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setError(null);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-2xl mx-auto"
    >
      <div className="bg-slate-800 rounded-2xl p-8 shadow-2xl border border-slate-700">
        <div className="text-center mb-6">
          <h2 className="text-3xl font-bold text-white mb-2">Upload Your CV</h2>
          <p className="text-slate-400">
            Support for PDF, DOCX, and TXT files (max 10MB)
          </p>
        </div>

        <div
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={clsx(
            'relative border-2 border-dashed rounded-xl transition-all duration-300',
            isDragging
              ? 'border-blue-500 bg-blue-500/10'
              : 'border-slate-600 bg-slate-900/50 hover:border-slate-500',
            isAnalyzing && 'opacity-50 pointer-events-none'
          )}
        >
          <input
            type="file"
            id="cv-upload"
            accept=".pdf,.docx,.doc,.txt"
            onChange={handleFileInput}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={isAnalyzing}
          />

          <div className="py-12 px-6 text-center">
            <AnimatePresence mode="wait">
              {!selectedFile ? (
                <motion.div
                  key="upload-prompt"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <Upload className="w-16 h-16 mx-auto mb-4 text-slate-400" />
                  <p className="text-lg text-slate-300 mb-2">
                    Drag and drop your CV here
                  </p>
                  <p className="text-sm text-slate-500">or click to browse</p>
                </motion.div>
              ) : (
                <motion.div
                  key="file-selected"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="flex items-center justify-between bg-slate-800 rounded-lg p-4"
                >
                  <div className="flex items-center gap-3">
                    <FileText className="w-10 h-10 text-blue-400" />
                    <div className="text-left">
                      <p className="text-white font-medium">{selectedFile.name}</p>
                      <p className="text-sm text-slate-400">
                        {(selectedFile.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={handleRemoveFile}
                    className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
                    disabled={isAnalyzing}
                  >
                    <X className="w-5 h-5 text-slate-400" />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            {isAnalyzing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-4 flex items-center justify-center gap-2 text-blue-400"
              >
                <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-400 border-t-transparent" />
                <span>Analyzing your CV...</span>
              </motion.div>
            )}
          </div>
        </div>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-4 bg-red-500/10 border border-red-500/50 rounded-lg text-red-400 text-sm"
          >
            {error}
          </motion.div>
        )}

        <div className="mt-6 flex items-center gap-2 text-sm text-slate-400">
          <CheckCircle2 className="w-4 h-4 text-green-400" />
          <span>Secure processing - Your data is never stored</span>
        </div>
      </div>
    </motion.div>
  );
};
