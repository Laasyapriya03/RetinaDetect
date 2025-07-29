const BACKEND_BASE_URL = 'http://localhost:8000';

export interface CodeChunk {
  content: string;
  source: string;
  start_line: number;
  type: string;
  distance?: number;
  function_name?: string;
}

export interface QueryResponse {
  answer: string;
  retrieved_context: CodeChunk[];
  debug_info: {
    retrieved_chunk_count: number;
    query_top_k: number;
    query_similarity_threshold: number;
    llm_temperature: number;
  };
}

export interface QueryRequest {
  query: string;
  temperature?: number;
  top_k?: number;
  similarity_threshold?: number;
}

class BackendApiService {
  private async request(endpoint: string, options: RequestInit = {}) {
    const response = await fetch(`${BACKEND_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorData}`);
    }

    return response.json();
  }

  async uploadCodeFile(file: File): Promise<{ message: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${BACKEND_BASE_URL}/upload_code_file`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(`Upload failed: ${errorData}`);
    }

    return response.json();
  }

  async clearCodebase(): Promise<{ message: string }> {
    return this.request('/clear_codebase', {
      method: 'POST',
    });
  }

  async askQuestion(request: QueryRequest): Promise<QueryResponse> {
    return this.request('/ask/', {
      method: 'POST',
      body: JSON.stringify({
        query: request.query,
        temperature: request.temperature || 0.2,
        top_k: request.top_k || 5,
        similarity_threshold: request.similarity_threshold || 0.7,
      }),
    });
  }

  async checkHealth(): Promise<{ message: string }> {
    return this.request('/');
  }
}

export const backendApi = new BackendApiService();