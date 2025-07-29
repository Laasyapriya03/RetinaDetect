import React, { useState, useEffect } from 'react';
import { BackendFileUpload } from './BackendFileUpload';
import { InteractiveChat } from './InteractiveChat';
import { UploadedFiles } from './UploadedFiles';
import { CodeDiagrams } from './CodeDiagrams';
import { Header } from './Header';
import { FileData } from '../services/api';
import { backendApi } from '../services/backendApi';
import { useDarkMode } from '../context/DarkModeContext';
import { Server, MessageSquare, FileText, BarChart3, AlertCircle, CheckCircle } from 'lucide-react';

export const EnhancedDashboard: React.FC = () => {
  const [hasBackendFiles, setHasBackendFiles] = useState(false);
  const [localFiles, setLocalFiles] = useState<FileData[]>([]);
  const [loading, setLoading] = useState(true);
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const [activeTab, setActiveTab] = useState<'chat' | 'files' | 'diagrams'>('chat');
  const { isDarkMode } = useDarkMode();

  const themeClasses = {
    background: isDarkMode 
      ? 'bg-slate-900' 
      : 'bg-gradient-to-br from-cyan-50 to-sky-100',
    text: isDarkMode ? 'text-white' : 'text-slate-800',
    cardBg: isDarkMode 
      ? 'bg-gray-800 border-gray-700' 
      : 'bg-white/80 border-cyan-200',
    tabActive: isDarkMode 
      ? 'bg-cyan-600 text-white border-cyan-600' 
      : 'bg-cyan-500 text-white border-cyan-500',
    tabInactive: isDarkMode 
      ? 'bg-gray-700 text-gray-300 hover:bg-gray-600 border-gray-600' 
      : 'bg-gray-200 text-gray-700 hover:bg-gray-300 border-gray-300',
    statusConnected: isDarkMode ? 'text-green-400' : 'text-green-600',
    statusDisconnected: isDarkMode ? 'text-red-400' : 'text-red-600',
    gradientText: 'bg-gradient-to-r from-cyan-400 via-sky-400 to-teal-400',
    border: isDarkMode ? 'border-gray-700' : 'border-cyan-200'
  };

  useEffect(() => {
    checkBackendStatus();
    checkLocalFiles();
  }, []);

  const checkBackendStatus = async () => {
    try {
      await backendApi.checkHealth();
      setBackendStatus('connected');
    } catch (error) {
      setBackendStatus('disconnected');
    }
  };

  const checkLocalFiles = async () => {
    try {
      const savedFiles = localStorage.getItem('uploadedFiles');
      if (savedFiles) {
        const parsedFiles = JSON.parse(savedFiles);
        setLocalFiles(parsedFiles);
      }
    } catch (error) {
      console.error('Error checking local files:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleBackendFilesUploaded = async (files: File[]) => {
    setHasBackendFiles(true);
    
    // Also save to localStorage for local features
    try {
      const fileData: FileData[] = await Promise.all(
        files.map(async (file) => ({
          name: file.name,
          content: await file.text(),
          type: file.name.split('.').pop() || 'unknown'
        }))
      );

      const existingFiles = JSON.parse(localStorage.getItem('uploadedFiles') || '[]');
      const newFiles = fileData.filter(newFile => 
        !existingFiles.some((existing: FileData) => existing.name === newFile.name)
      );
      
      const updatedFiles = [...existingFiles, ...newFiles];
      localStorage.setItem('uploadedFiles', JSON.stringify(updatedFiles));
      setLocalFiles(updatedFiles);
    } catch (error) {
      console.error('Error saving files locally:', error);
    }
  };

  const handleClearBackendFiles = () => {
    setHasBackendFiles(false);
  };

  const tabs = [
    { id: 'chat' as const, label: 'AI Chat', icon: MessageSquare },
    { id: 'files' as const, label: 'File Manager', icon: FileText },
    { id: 'diagrams' as const, label: 'Code Diagrams', icon: BarChart3 }
  ];

  if (loading) {
    return (
      <div className={`min-h-screen ${themeClasses.background} ${themeClasses.text} flex items-center justify-center transition-colors duration-300`}>
        <div className="animate-spin h-12 w-12 border-4 border-cyan-400 border-t-transparent rounded-full"></div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen ${themeClasses.background} ${themeClasses.text} transition-colors duration-300`}>
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* Status Bar */}
        <div className={`mb-6 p-4 rounded-lg ${themeClasses.cardBg} border ${themeClasses.border} flex items-center justify-between`}>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Server className="h-5 w-5" />
              <span className="font-medium">Backend Status:</span>
              <span className={`flex items-center space-x-1 ${
                backendStatus === 'connected' ? themeClasses.statusConnected : themeClasses.statusDisconnected
              }`}>
                {backendStatus === 'connected' ? (
                  <CheckCircle className="h-4 w-4" />
                ) : (
                  <AlertCircle className="h-4 w-4" />
                )}
                <span className="capitalize">{backendStatus}</span>
              </span>
            </div>
            {hasBackendFiles && (
              <div className={`flex items-center space-x-2 ${themeClasses.statusConnected}`}>
                <CheckCircle className="h-4 w-4" />
                <span className="text-sm">Files loaded in backend</span>
              </div>
            )}
          </div>
          <button
            onClick={checkBackendStatus}
            className="px-3 py-1 text-sm bg-cyan-500 hover:bg-cyan-600 text-white rounded transition-colors"
          >
            Refresh Status
          </button>
        </div>

        {/* Upload Section */}
        {!hasBackendFiles && (
          <div className="flex flex-col items-center justify-center mb-8">
            <div className="text-center mb-8">
              <h1 className={`text-4xl font-bold mb-4 ${themeClasses.gradientText} bg-clip-text text-transparent`}>
                Interactive Code Analysis
              </h1>
              <p className={`text-lg max-w-2xl mx-auto opacity-80`}>
                Upload your C/C++ files to the backend for AI-powered analysis and interactive chat
              </p>
            </div>
            <BackendFileUpload 
              onFilesUploaded={handleBackendFilesUploaded}
              onClearFiles={handleClearBackendFiles}
            />
          </div>
        )}

        {/* Main Content */}
        {hasBackendFiles && (
          <div className="space-y-6">
            {/* Tab Navigation */}
            <div className={`flex space-x-1 p-1 rounded-lg ${themeClasses.cardBg} border ${themeClasses.border}`}>
              {tabs.map(tab => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-md font-medium transition-all ${
                      activeTab === tab.id ? themeClasses.tabActive : themeClasses.tabInactive
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Tab Content */}
            <div className="min-h-[600px]">
              {activeTab === 'chat' && (
                <InteractiveChat hasFiles={hasBackendFiles} />
              )}
              
              {activeTab === 'files' && (
                <div className="space-y-6">
                  <UploadedFiles isDarkMode={isDarkMode} />
                  <div className={`p-6 rounded-lg ${themeClasses.cardBg} border ${themeClasses.border}`}>
                    <h2 className="text-xl font-semibold mb-4">Upload More Files</h2>
                    <BackendFileUpload 
                      onFilesUploaded={handleBackendFilesUploaded}
                      onClearFiles={handleClearBackendFiles}
                    />
                  </div>
                </div>
              )}
              
              {activeTab === 'diagrams' && (
                <CodeDiagrams files={localFiles} isDarkMode={isDarkMode} />
              )}
            </div>

            {/* Clear All Button */}
            <div className={`flex justify-center pt-6 border-t ${themeClasses.border}`}>
              <button
                onClick={() => {
                  handleClearBackendFiles();
                  localStorage.removeItem('uploadedFiles');
                  setLocalFiles([]);
                }}
                className="px-6 py-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors"
              >
                Clear All Data
              </button>
            </div>
          </div>
        )}

        {/* Backend Connection Error */}
        {backendStatus === 'disconnected' && (
          <div className={`mt-8 p-6 rounded-lg border border-red-300 bg-red-100 dark:bg-red-900/20 dark:border-red-700`}>
            <div className="flex items-center space-x-2 mb-2">
              <AlertCircle className="h-5 w-5 text-red-500" />
              <h3 className="font-semibold text-red-800 dark:text-red-200">Backend Connection Failed</h3>
            </div>
            <p className="text-red-700 dark:text-red-300 text-sm mb-3">
              Cannot connect to the FastAPI backend at http://localhost:8000
            </p>
            <div className="text-sm text-red-600 dark:text-red-400">
              <p className="font-medium mb-1">To start the backend:</p>
              <ol className="list-decimal list-inside space-y-1 font-mono text-xs">
                <li>cd backend/</li>
                <li>pip install -r requirements.txt</li>
                <li>python main.py</li>
              </ol>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};