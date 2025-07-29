import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Code, Clock, Brain, Search, AlertCircle, CheckCircle } from 'lucide-react';
import { backendApi, QueryResponse, CodeChunk } from '../services/backendApi';
import { useDarkMode } from '../context/DarkModeContext';
import { MermaidDiagram } from './MermaidDiagram';

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  response?: QueryResponse;
  isLoading?: boolean;
}

interface InteractiveChatProps {
  hasFiles: boolean;
}

export const InteractiveChat: React.FC<InteractiveChatProps> = ({ hasFiles }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [settings, setSettings] = useState({
    temperature: 0.2,
    top_k: 5,
    similarity_threshold: 0.7,
  });
  const [showSettings, setShowSettings] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { isDarkMode } = useDarkMode();

  const themeClasses = {
    container: isDarkMode ? 'bg-gray-900' : 'bg-gray-50',
    chatBg: isDarkMode ? 'bg-gray-800' : 'bg-white',
    userMessage: isDarkMode ? 'bg-cyan-600' : 'bg-cyan-500',
    botMessage: isDarkMode ? 'bg-gray-700' : 'bg-gray-100',
    text: isDarkMode ? 'text-white' : 'text-gray-900',
    secondaryText: isDarkMode ? 'text-gray-300' : 'text-gray-600',
    border: isDarkMode ? 'border-gray-600' : 'border-gray-200',
    inputBg: isDarkMode ? 'bg-gray-700' : 'bg-white',
    buttonPrimary: isDarkMode 
      ? 'bg-cyan-600 hover:bg-cyan-700' 
      : 'bg-cyan-500 hover:bg-cyan-600',
    settingsBg: isDarkMode ? 'bg-gray-700' : 'bg-gray-100',
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading || !hasFiles) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    const botMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'bot',
      content: '',
      timestamp: new Date(),
      isLoading: true,
    };

    setMessages(prev => [...prev, userMessage, botMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await backendApi.askQuestion({
        query: userMessage.content,
        temperature: settings.temperature,
        top_k: settings.top_k,
        similarity_threshold: settings.similarity_threshold,
      });

      setMessages(prev => prev.map(msg => 
        msg.id === botMessage.id 
          ? { ...msg, content: response.answer, response, isLoading: false }
          : msg
      ));
    } catch (error) {
      setMessages(prev => prev.map(msg => 
        msg.id === botMessage.id 
          ? { 
              ...msg, 
              content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
              isLoading: false 
            }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const renderCodeChunk = (chunk: CodeChunk, index: number) => (
    <div key={index} className={`p-3 rounded-lg border ${themeClasses.border} ${themeClasses.settingsBg} mb-2`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <Code className="h-4 w-4 text-cyan-500" />
          <span className={`text-sm font-medium ${themeClasses.text}`}>
            {chunk.source} (Line {chunk.start_line})
          </span>
        </div>
        <div className="flex items-center space-x-2 text-xs">
          <span className={`px-2 py-1 rounded ${chunk.type === 'code' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'}`}>
            {chunk.type}
          </span>
          {chunk.distance && (
            <span className={`px-2 py-1 rounded bg-gray-100 text-gray-800`}>
              {chunk.distance.toFixed(3)}
            </span>
          )}
        </div>
      </div>
      {chunk.function_name && (
        <div className={`text-sm ${themeClasses.secondaryText} mb-2`}>
          Function: <code className="font-mono">{chunk.function_name}</code>
        </div>
      )}
      <pre className={`text-sm ${themeClasses.secondaryText} overflow-x-auto font-mono bg-black/10 p-2 rounded`}>
        {chunk.content}
      </pre>
    </div>
  );

  const extractMermaidDiagram = (content: string) => {
    const mermaidRegex = /```mermaid\n([\s\S]*?)\n```/;
    const match = content.match(mermaidRegex);
    if (match) {
      return {
        diagram: match[1].trim(),
        textWithoutDiagram: content.replace(mermaidRegex, '').trim()
      };
    }
    return { diagram: null, textWithoutDiagram: content };
  };

  if (!hasFiles) {
    return (
      <div className={`${themeClasses.chatBg} rounded-lg p-8 text-center border ${themeClasses.border}`}>
        <Bot className={`h-16 w-16 ${themeClasses.secondaryText} mx-auto mb-4 opacity-50`} />
        <h3 className={`text-xl font-semibold mb-2 ${themeClasses.text}`}>Upload Code Files First</h3>
        <p className={themeClasses.secondaryText}>
          Upload your C/C++ files to start chatting with your codebase.
        </p>
      </div>
    );
  }

  return (
    <div className={`${themeClasses.chatBg} rounded-lg border ${themeClasses.border} flex flex-col h-[600px]`}>
      {/* Header */}
      <div className={`p-4 border-b ${themeClasses.border} flex items-center justify-between`}>
        <div className="flex items-center space-x-2">
          <Bot className="h-6 w-6 text-cyan-500" />
          <h2 className={`text-lg font-semibold ${themeClasses.text}`}>Code Assistant</h2>
        </div>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className={`px-3 py-1 text-sm rounded ${themeClasses.buttonPrimary} text-white transition-colors`}
        >
          Settings
        </button>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className={`p-4 border-b ${themeClasses.border} ${themeClasses.settingsBg}`}>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className={`block text-sm font-medium ${themeClasses.text} mb-1`}>
                Temperature: {settings.temperature}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={settings.temperature}
                onChange={(e) => setSettings(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>
            <div>
              <label className={`block text-sm font-medium ${themeClasses.text} mb-1`}>
                Top K: {settings.top_k}
              </label>
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={settings.top_k}
                onChange={(e) => setSettings(prev => ({ ...prev, top_k: parseInt(e.target.value) }))}
                className="w-full"
              />
            </div>
            <div>
              <label className={`block text-sm font-medium ${themeClasses.text} mb-1`}>
                Similarity: {settings.similarity_threshold}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={settings.similarity_threshold}
                onChange={(e) => setSettings(prev => ({ ...prev, similarity_threshold: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-8">
            <Bot className={`h-12 w-12 ${themeClasses.secondaryText} mx-auto mb-4 opacity-50`} />
            <p className={themeClasses.secondaryText}>
              Ask me anything about your uploaded code!
            </p>
            <div className="mt-4 space-y-2 text-sm">
              <p className={`${themeClasses.secondaryText} italic`}>Try asking:</p>
              <div className="space-y-1">
                <p className={`${themeClasses.secondaryText} text-xs`}>"What does the main function do?"</p>
                <p className={`${themeClasses.secondaryText} text-xs`}>"Explain the reverse_string function"</p>
                <p className={`${themeClasses.secondaryText} text-xs`}>"Show me all the functions that use loops"</p>
              </div>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
              <div className="flex items-center space-x-2 mb-1">
                {message.type === 'user' ? (
                  <User className="h-4 w-4 text-cyan-500" />
                ) : (
                  <Bot className="h-4 w-4 text-cyan-500" />
                )}
                <span className={`text-xs ${themeClasses.secondaryText}`}>
                  {message.timestamp.toLocaleTimeString()}
                </span>
              </div>
              
              <div className={`p-3 rounded-lg ${
                message.type === 'user' 
                  ? `${themeClasses.userMessage} text-white` 
                  : `${themeClasses.botMessage} ${themeClasses.text}`
              }`}>
                {message.isLoading ? (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin h-4 w-4 border-2 border-cyan-500 border-t-transparent rounded-full"></div>
                    <span>Thinking...</span>
                  </div>
                ) : (
                  <div>
                    {message.type === 'bot' && message.response ? (
                      <div>
                        {(() => {
                          const { diagram, textWithoutDiagram } = extractMermaidDiagram(message.content);
                          return (
                            <div>
                              <div className="whitespace-pre-wrap mb-4">{textWithoutDiagram}</div>
                              {diagram && (
                                <div className="mb-4">
                                  <h4 className={`font-semibold mb-2 ${themeClasses.text}`}>Generated Diagram:</h4>
                                  <MermaidDiagram chart={diagram} isDarkMode={isDarkMode} />
                                </div>
                              )}
                            </div>
                          );
                        })()}
                        
                        {/* Debug Info */}
                        <div className={`mt-3 p-2 rounded text-xs ${themeClasses.settingsBg} ${themeClasses.secondaryText}`}>
                          <div className="flex items-center space-x-4">
                            <span className="flex items-center space-x-1">
                              <Search className="h-3 w-3" />
                              <span>{message.response.debug_info.retrieved_chunk_count} chunks</span>
                            </span>
                            <span className="flex items-center space-x-1">
                              <Brain className="h-3 w-3" />
                              <span>T: {message.response.debug_info.llm_temperature}</span>
                            </span>
                            <span className="flex items-center space-x-1">
                              <Clock className="h-3 w-3" />
                              <span>K: {message.response.debug_info.query_top_k}</span>
                            </span>
                          </div>
                        </div>

                        {/* Retrieved Context */}
                        {message.response.retrieved_context.length > 0 && (
                          <details className="mt-3">
                            <summary className={`cursor-pointer text-sm font-medium ${themeClasses.text} hover:text-cyan-500`}>
                              View Retrieved Code Context ({message.response.retrieved_context.length} chunks)
                            </summary>
                            <div className="mt-2 space-y-2">
                              {message.response.retrieved_context.map((chunk, index) => 
                                renderCodeChunk(chunk, index)
                              )}
                            </div>
                          </details>
                        )}
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap">{message.content}</div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className={`p-4 border-t ${themeClasses.border}`}>
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about your code..."
            className={`flex-1 p-3 rounded-lg border ${themeClasses.border} ${themeClasses.inputBg} ${themeClasses.text} focus:outline-none focus:ring-2 focus:ring-cyan-500`}
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || !inputValue.trim()}
            className={`px-4 py-3 ${themeClasses.buttonPrimary} text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
};