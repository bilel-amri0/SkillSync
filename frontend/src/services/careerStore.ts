import type { CVAnalysisResponse } from '../api';
import type { CareerGuidanceResponse } from '../types/careerGuidance';

const ANALYSIS_KEY = 'skillsync.latestAnalysis';
const GUIDANCE_KEY = 'skillsync.latestGuidance';

interface StoredPayload<T> {
  savedAt: string;
  payload: T;
}

const readStorage = <T>(key: string): StoredPayload<T> | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) {
      return null;
    }
    return JSON.parse(raw) as StoredPayload<T>;
  } catch (error) {
    console.warn(`[careerStore] Failed to parse ${key}`, error);
    return null;
  }
};

const writeStorage = <T>(key: string, payload: T) => {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    const wrapped: StoredPayload<T> = {
      payload,
      savedAt: new Date().toISOString(),
    };
    window.localStorage.setItem(key, JSON.stringify(wrapped));
    window.dispatchEvent(new StorageEvent('storage', { key }));
  } catch (error) {
    console.warn(`[careerStore] Failed to persist ${key}`, error);
  }
};

export const saveLatestAnalysis = (analysis: CVAnalysisResponse) => {
  writeStorage(ANALYSIS_KEY, analysis);
};

export const saveLatestGuidance = (guidance: CareerGuidanceResponse) => {
  writeStorage(GUIDANCE_KEY, guidance);
};

export const loadLatestAnalysis = (): CVAnalysisResponse | null => {
  return readStorage<CVAnalysisResponse>(ANALYSIS_KEY)?.payload ?? null;
};

export const loadLatestGuidance = (): CareerGuidanceResponse | null => {
  return readStorage<CareerGuidanceResponse>(GUIDANCE_KEY)?.payload ?? null;
};

export const clearCareerSnapshots = () => {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.removeItem(ANALYSIS_KEY);
  window.localStorage.removeItem(GUIDANCE_KEY);
};
