import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

interface MermaidDiagramProps {
  chart: string;
  isDarkMode: boolean;
}

export const MermaidDiagram: React.FC<MermaidDiagramProps> = ({ chart, isDarkMode }) => {
  const elementRef = useRef<HTMLDivElement>(null);
  const [diagramId] = React.useState(() => `mermaid-${Math.random().toString(36).substr(2, 9)}`);

  useEffect(() => {
    mermaid.initialize({
      startOnLoad: true,
      theme: isDarkMode ? 'dark' : 'neutral',
      securityLevel: 'loose',
      fontFamily: 'Arial, sans-serif',
      fontSize: 16,
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis',
        padding: 20
      },
      class: {
        useMaxWidth: true
      },
      gitGraph: {
        useMaxWidth: true
      }
    });
  }, [isDarkMode]);

  useEffect(() => {
    const renderDiagram = async () => {
      if (elementRef.current && chart.trim()) {
        try {
          // Clear previous content
          elementRef.current.innerHTML = '';
          
          // Create a temporary div for mermaid
          const tempDiv = document.createElement('div');
          tempDiv.innerHTML = `<div class="mermaid">${chart}</div>`;
          
          // Render the diagram
          const { svg } = await mermaid.render(diagramId, chart);
          elementRef.current.innerHTML = svg;
        } catch (error) {
          console.error('Mermaid rendering error:', error);
          elementRef.current.innerHTML = `
            <div class="p-4 text-center text-red-500">
              <p>Error rendering diagram</p>
              <pre class="text-xs mt-2 text-left overflow-auto">${chart}</pre>
            </div>
          `;
        }
      }
    };

    renderDiagram();
  }, [chart, isDarkMode, diagramId]);

  if (!chart.trim()) {
    return (
      <div className="flex items-center justify-center p-8 text-gray-500">
        <p>No diagram data available</p>
      </div>
    );
    }

  return (
    <div 
      ref={elementRef} 
      className={`mermaid-diagram w-full overflow-auto p-4 rounded-lg border ${
        isDarkMode ? 'bg-gray-800 border-gray-600' : 'bg-white border-gray-200'
      }`}
      style={{ minHeight: '300px' }}
    />
  );
};