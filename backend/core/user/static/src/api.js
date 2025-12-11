// Global fetch wrapper for API requests with rate-limit (429) handling
// This file must be loaded before other site scripts so that window.fetch is patched

(function() {
  if (window.__API_FETCH_PATCHED__) return; // Prevent double patch
  window.__API_FETCH_PATCHED__ = true;

  const originalFetch = window.fetch;

  window.fetch = async function(input, init) {
    const response = await originalFetch(input, init);
    if (response.status === 429) {
      let retryAfter = Number(response.headers.get('Retry-After')) || 0;
      try {
        const data = await response.clone().json();
        retryAfter = data.retry_after || retryAfter;
      } catch (_) {}
      const T = window.I18N_UI || {};
      const title = T.rateLimitTitle || 'Rate Limit Exceeded';
      const prefix = T.rateLimitPrefix || "You've made too many requests. Please wait";
      const suffix = T.rateLimitSuffix || 'seconds before trying again.';
      window.showModal?.({
        title,
        message: `${prefix} ${retryAfter || 60} ${suffix}`,
        variant: 'warning'
      });
      // Still reject so caller can handle if needed
      const error = new Error('Rate limit exceeded');
      error.rateLimited = true;
      error.retryAfter = retryAfter;
      throw error;
    }

    if (!response.ok) {
      // Generic error modal
      try {
        const text = await response.clone().text();
        const T = window.I18N_UI || {};
        const title = T.errorTitle || 'Error';
        window.showModal?.({ title, message: text.slice(0, 140), variant: 'error' });
      } catch (_) {}
    }
    return response;
  };
})();
