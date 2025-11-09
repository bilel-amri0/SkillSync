import { useState, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  FileText, 
  Download, 
  Eye, 
  CheckCircle, 
  AlertCircle, 
  Loader,
  X,
  Plus,
  Award,
  Briefcase,
  GraduationCap,
  ExternalLink
} from 'lucide-react';
import { cvApi, handleApiError } from '../services/api';
import toast from 'react-hot-toast';
import type { CVAnalysis } from '../types';

const CVAnalysis = () => {
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [analysisResults, setAnalysisResults] = useState<CVAnalysis[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  const { data: analyses, isLoading, error } = useQuery({
    queryKey: ['cv-analyses'],
    queryFn: async () => {
      // This would fetch CV analyses from API when available
      // For now, return empty array until backend is connected
      return [];
    },
  });

  // Show error state if API call fails
  if (error) {
    toast.error('Unable to connect to backend server. Please ensure your FastAPI server is running on localhost:8000.');
  }

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const validFiles = acceptedFiles.filter(file => {
      const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
      const maxSize = 10 * 1024 * 1024; // 10MB

      if (!validTypes.includes(file.type)) {
        toast.error(`${file.name} is not a valid file type. Please upload PDF or DOCX files only.`);
        return false;
      }

      if (file.size > maxSize) {
        toast.error(`${file.name} is too large. Maximum file size is 10MB.`);
        return false;
      }

      return true;
    });

    setUploadedFiles(prev => [...prev, ...validFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    multiple: true
  } as any);

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const uploadAndAnalyze = async () => {
    if (uploadedFiles.length === 0) {
      toast.error('Please select files to upload');
      return;
    }

    setIsUploading(true);

    try {
      for (const file of uploadedFiles) {
        // Upload to backend
        const response = await cvApi.upload(file);
        const cvId = response.data.id;
        
        // Get analysis results
        const analysisResponse = await cvApi.analyze(cvId);
        setAnalysisResults(prev => [...prev, analysisResponse.data]);
        
        toast.success(`${file.name} uploaded and analyzed successfully!`);
      }

      setUploadedFiles([]);
    } catch (error) {
      const errorMessage = handleApiError(error);
      toast.error(`Upload failed: ${errorMessage}`);
    } finally {
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getSkillLevelColor = (level: number) => {
    if (level >= 8) return 'text-green-600 bg-green-100';
    if (level >= 6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const allAnalyses = analyses || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">CV Analysis</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Upload your CV to extract skills, experience, and get personalized insights
        </p>
      </motion.div>

      {/* Backend Connection Status */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4"
        >
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400 mr-3" />
            <div>
              <h3 className="text-sm font-medium text-red-800 dark:text-red-200">
                Backend Server Not Available
              </h3>
              <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                Cannot connect to your FastAPI backend at localhost:8000. Please ensure your backend server is running.
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Upload Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700"
      >
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Upload CV</h2>
        
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
              : 'border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500'
          }`}
        >
          <input {...(getInputProps() as any)} />
          <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          {isDragActive ? (
            <p className="text-blue-600 dark:text-blue-400">Drop the files here...</p>
          ) : (
            <div>
              <p className="text-gray-600 dark:text-gray-400 mb-2">
                Drag & drop CV files here, or click to select
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-500">
                Supports PDF and DOCX files (max 10MB)
              </p>
            </div>
          )}
        </div>

        {/* Uploaded Files */}
        <AnimatePresence>
          {uploadedFiles.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 space-y-2"
            >
              {uploadedFiles.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <FileText className="h-5 w-5 text-gray-500" />
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">{file.name}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">{formatFileSize(file.size)}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => removeFile(index)}
                    className="text-red-500 hover:text-red-700 transition-colors"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Upload Button */}
        {uploadedFiles.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="mt-4 flex justify-center"
          >
            <button
              onClick={uploadAndAnalyze}
              disabled={isUploading || !!error}
              className="flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isUploading ? (
                <>
                  <Loader className="animate-spin h-4 w-4 mr-2" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4 mr-2" />
                  Upload & Analyze CV
                </>
              )}
            </button>
          </motion.div>
        )}
      </motion.div>

      {/* Analysis Results */}
      <AnimatePresence>
        {allAnalyses.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Analysis Results</h2>
            
            {allAnalyses.map((analysis, index) => (
              <motion.div
                key={analysis.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700"
              >
                {/* Header */}
                <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <FileText className="h-8 w-8 text-blue-600" />
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                          {analysis.file_name}
                        </h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                          Uploaded {new Date(analysis.upload_date).toLocaleDateString()} • {formatFileSize(analysis.file_size)}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-5 w-5 text-green-500" />
                      <span className="text-sm font-medium text-green-600">Analyzed</span>
                    </div>
                  </div>
                </div>

                <div className="p-6 space-y-6">
                  {/* Summary */}
                  <div>
                    <h4 className="text-md font-medium text-gray-900 dark:text-white mb-2">Summary</h4>
                    <p className="text-gray-600 dark:text-gray-400">{analysis.summary}</p>
                  </div>

                  {/* Skills */}
                  <div>
                    <h4 className="text-md font-medium text-gray-900 dark:text-white mb-3">Extracted Skills</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {analysis.skills.map((skill, skillIndex) => (
                        <div
                          key={skillIndex}
                          className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                        >
                          <div className="flex-1">
                            <div className="flex items-center space-x-2">
                              <span className="font-medium text-gray-900 dark:text-white">{skill.name}</span>
                              <span className="text-xs text-gray-500 dark:text-gray-400">({skill.category})</span>
                            </div>
                            <div className="mt-1 flex items-center space-x-2">
                              <div className="flex-1 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                                <div
                                  className="bg-blue-600 h-2 rounded-full"
                                  style={{ width: `${skill.proficiency_level * 10}%` }}
                                />
                              </div>
                              <span className="text-xs text-gray-500 dark:text-gray-400">
                                {skill.proficiency_level}/10
                              </span>
                            </div>
                          </div>
                          <div className="ml-3">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSkillLevelColor(skill.proficiency_level)}`}>
                              {skill.confidence}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Experience */}
                  {analysis.experience.length > 0 && (
                    <div>
                      <h4 className="text-md font-medium text-gray-900 dark:text-white mb-3">Experience</h4>
                      <div className="space-y-3">
                        {analysis.experience.map((exp, expIndex) => (
                          <div key={expIndex} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                            <div className="flex items-start space-x-3">
                              <Briefcase className="h-5 w-5 text-gray-500 mt-1" />
                              <div className="flex-1">
                                <h5 className="font-medium text-gray-900 dark:text-white">{exp.position}</h5>
                                <p className="text-sm text-gray-600 dark:text-gray-400">{exp.company}</p>
                                <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                                  {new Date(exp.start_date).toLocaleDateString()} - 
                                  {exp.end_date ? new Date(exp.end_date).toLocaleDateString() : 'Present'}
                                  {' • '}{Math.round(exp.duration_months / 12 * 10) / 10} years
                                </p>
                                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">{exp.description}</p>
                                {exp.achievements.length > 0 && (
                                  <div className="mt-2">
                                    <p className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">Key Achievements:</p>
                                    <ul className="text-xs text-gray-600 dark:text-gray-400 list-disc list-inside">
                                      {exp.achievements.map((achievement, achIndex) => (
                                        <li key={achIndex}>{achievement}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Education */}
                  {analysis.education.length > 0 && (
                    <div>
                      <h4 className="text-md font-medium text-gray-900 dark:text-white mb-3">Education</h4>
                      <div className="space-y-3">
                        {analysis.education.map((edu, eduIndex) => (
                          <div key={eduIndex} className="flex items-start space-x-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                            <GraduationCap className="h-5 w-5 text-gray-500 mt-1" />
                            <div>
                              <h5 className="font-medium text-gray-900 dark:text-white">{edu.degree}</h5>
                              <p className="text-sm text-gray-600 dark:text-gray-400">{edu.institution}</p>
                              <p className="text-xs text-gray-500 dark:text-gray-500">
                                {edu.field_of_study} • 
                                {new Date(edu.start_date).getFullYear()} - 
                                {edu.end_date ? new Date(edu.end_date).getFullYear() : 'Present'}
                                {edu.gpa && ` • GPA: ${edu.gpa}`}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Certifications */}
                  {analysis.certifications.length > 0 && (
                    <div>
                      <h4 className="text-md font-medium text-gray-900 dark:text-white mb-3">Certifications</h4>
                      <div className="space-y-3">
                        {analysis.certifications.map((cert, certIndex) => (
                          <div key={certIndex} className="flex items-start space-x-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                            <Award className="h-5 w-5 text-gray-500 mt-1" />
                            <div className="flex-1">
                              <h5 className="font-medium text-gray-900 dark:text-white">{cert.name}</h5>
                              <p className="text-sm text-gray-600 dark:text-gray-400">{cert.issuer}</p>
                              <p className="text-xs text-gray-500 dark:text-gray-500">
                                Earned {new Date(cert.date_earned).toLocaleDateString()}
                                {cert.credential_id && ` • ID: ${cert.credential_id}`}
                              </p>
                            </div>
                            {cert.verification_url && (
                              <ExternalLink className="h-4 w-4 text-gray-400 hover:text-gray-600 cursor-pointer" />
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty State */}
      {allAnalyses.length === 0 && !isLoading && !error && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-gray-800 p-12 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 text-center"
        >
          <FileText className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No CVs analyzed yet
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Upload your first CV to start analyzing your skills and experience
          </p>
          <button
            onClick={() => (document.querySelector('input[type="file"]') as HTMLInputElement)?.click()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Upload Your First CV
          </button>
        </motion.div>
      )}
    </div>
  );
};

export default CVAnalysis;