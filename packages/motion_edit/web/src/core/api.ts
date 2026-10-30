export class ApiError extends Error {
  constructor(public readonly status: number, message: string) { super(message); }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...init });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message = Array.isArray(detail.detail)
      ? detail.detail.map((item: { loc?: string[]; msg?: string }) => `${item.loc?.join('.') || 'Request'}: ${item.msg || 'Invalid value'}`).join('\n')
      : typeof detail.detail === 'string' ? detail.detail : response.statusText;
    throw new ApiError(response.status, message || `HTTP ${response.status}`);
  }
  return response.json();
}
