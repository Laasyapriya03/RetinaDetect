import React, { useCallback, useState } from 'react';
import { Upload, File, CheckCircle, AlertCircle } from 'lucide-react';
import { useDarkMode } from '../context/DarkModeContext';

interface FileUploadProps {
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFilesUploaded }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string>('');
  const { isDarkMode } = useDarkMode();

  const themeClasses = {
    container: isDarkMode ? 'bg-gray-800' : 'bg-white',
    card: {
      base: isDarkMode 
        ? 'border-gray-600 bg-gray-700' 
        : 'border-cyan-300 bg-white',
      active: isDarkMode 
        ? 'border-blue-400 bg-blue-900/20' 
        : 'border-cyan-500 bg-cyan-100/50',
      disabled: 'opacity-50 pointer-events-none'
    },
    text: {
      primary: isDarkMode ? 'text-white' : 'text-slate-800',
      secondary: isDarkMode ? 'text-gray-300' : 'text-slate-600',
      tertiary: isDarkMode ? 'text-gray-400' : 'text-slate-500',
    },
    error: isDarkMode 
      ? 'bg-red-900/80 border-red-700 text-red-100' 
      : 'bg-red-100 border-red-300 text-red-800',
    button: {
      primary: isDarkMode 
        ? 'bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-700 hover:to-teal-700' 
        : 'bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-600 hover:to-teal-600',
      icon: isDarkMode ? 'text-gray-300' : 'text-slate-500'
    },
    spinner: isDarkMode 
      ? 'border-cyan-400 border-t-transparent' 
      : 'border-cyan-600 border-t-transparent'
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setIsProcessing(true);
    setError('');

    const files = Array.from(e.dataTransfer.files);
    const cppFiles = files.filter(file => 
      /\.(c|cpp|h|hpp)$/i.test(file.name)
    );

    if (cppFiles.length > 0) {
      try {
        await onFilesUploaded(cppFiles);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Upload failed');
      }
    } else {
      setError('Please upload C/C++ files (.c, .cpp, .h, .hpp)');
    }
    
    setIsProcessing(false);
  }, [onFilesUploaded]);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      setIsProcessing(true);
      setError('');
      try {
        await onFilesUploaded(files);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Upload failed');
      }
      setIsProcessing(false);
    }
  }, [onFilesUploaded]);

  return (
    <div className={`w-full max-w-2xl rounded-lg p-4 ${themeClasses.container}`}>
      {error && (
        <div className={`mb-4 p-4 ${themeClasses.error} border rounded-lg flex items-center`}>
          <AlertCircle className="h-5 w-5 mr-2" />
          {error}
        </div>
      )}
      
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300 ${
          isDragging
            ? themeClasses.card.active
            : themeClasses.card.base
        } ${isProcessing ? themeClasses.card.disabled : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isProcessing ? (
          <div className="space-y-4">
            <div className={`animate-spin h-12 w-12 border-4 ${themeClasses.spinner} rounded-full mx-auto`}></div>
            <p className={themeClasses.text.secondary}>Uploading and processing your C/C++ files...</p>
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className={`h-16 w-16 mx-auto ${themeClasses.button.icon}`} />
            <div>
              <h3 className={`text-xl font-semibold mb-2 ${themeClasses.text.primary}`}>
                Upload Your C/C++ Files
              </h3>
              <p className={`${themeClasses.text.secondary} mb-4`}>
                Drag and drop your .c, .cpp, .h, .hpp files here, or click to browse
              </p>
              <p className={`text-sm ${themeClasses.text.tertiary} mb-6`}>
                Supported formats: .c, .cpp, .h, .hpp
              </p>
            </div>
            <label className={`inline-flex items-center px-6 py-3 ${themeClasses.button.primary} rounded-lg cursor-pointer transition-all text-white shadow-lg hover:scale-105`}>
              <File className="h-5 w-5 mr-2" />
              Browse Files
              <input
                type="file"
                multiple
                accept=".c,.cpp,.h,.hpp"
                className="hidden"
                onChange={handleFileSelect}
              />
            </label>
          </div>
        )}
      </div>
    </div>
  );
};