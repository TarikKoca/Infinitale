// Demo Video Popup
// Site-wide confirmation modal
// Usage: await window.showConfirm({ title, message, confirmText, cancelText, variant })
// Returns a Promise<boolean>
try {
  if (!window.showConfirm) {
    window.showConfirm = function({ title = (window.I18N_UI?.areYouSure || 'Are you sure?'), message = '', confirmText = (window.I18N_UI?.confirm || 'Confirm'), cancelText = (window.I18N_UI?.cancel || 'Cancel'), variant = 'danger' } = {}) {
      return new Promise(resolve => {
        const backdrop = document.createElement('div');
        backdrop.className = 'fixed inset-0 z-[9999] flex items-center justify-center p-4';
        backdrop.style.background = 'rgba(0,0,0,0.6)';
        backdrop.style.transition = 'opacity 180ms ease';
        backdrop.setAttribute('role', 'dialog');
        backdrop.setAttribute('aria-modal', 'true');
        backdrop.innerHTML = `
          <div class="rounded-2xl shadow-2xl max-w-md w-[92vw] p-6" style="background: var(--bg-card); color: var(--text-primary); border: 1px solid var(--border-default);">
            <div class="flex items-start justify-between">
              <h3 class="text-xl font-bold">${title}</h3>
              <button class="modal-close-btn" aria-label="${(window.I18N_UI?.close || 'Close')}" data-close style="color: var(--text-secondary);">
                <i class="fa-solid fa-xmark text-xl"></i>
              </button>
            </div>
            ${message ? `<p class="mt-3" style="color: var(--text-secondary);">${message}</p>` : ''}
            <div class="mt-5 flex gap-3 justify-end">
              <button class="rounded-lg font-semibold" data-cancel style="padding: 0.5rem 1rem; background: var(--bg-card); color: var(--text-primary); border: 1px solid var(--border-default);">${cancelText}</button>
              <button class="rounded-lg font-semibold" data-confirm style="padding: 0.5rem 1rem; color: #fff; ${variant === 'danger' ? 'background: var(--color-error-600);' : 'background: var(--color-primary-600);'}">${confirmText}</button>
            </div>
          </div>
        `;

        function cleanup(result) {
          try {
            document.removeEventListener('keydown', onKeyDown);
            backdrop.removeEventListener('click', onBackdropClick);
          } catch (_) {}
          backdrop.style.opacity = '0';
          setTimeout(() => backdrop.remove(), 180);
          resolve(result);
        }

        function onBackdropClick(e) {
          if (e.target === backdrop) cleanup(false);
        }

        function onKeyDown(e) {
          if (e.key === 'Escape') cleanup(false);
          if (e.key === 'Enter') cleanup(true);
        }

        document.addEventListener('keydown', onKeyDown);
        backdrop.addEventListener('click', onBackdropClick);

        document.body.appendChild(backdrop);

        const cancelBtn = backdrop.querySelector('[data-cancel]');
        const confirmBtn = backdrop.querySelector('[data-confirm]');
        const closeBtn = backdrop.querySelector('[data-close]');

        // Setup hover styles using CSS variables
        function setupHover(el, base, hover) {
          if (!el) return;
          Object.assign(el.style, base);
          el.addEventListener('mouseenter', () => Object.assign(el.style, hover));
          el.addEventListener('mouseleave', () => Object.assign(el.style, base));
        }

        const cancelBase = { background: 'var(--bg-card)', color: 'var(--text-primary)', border: '1px solid var(--border-default)' };
        const cancelHover = { background: 'var(--bg-card-hover)', border: '1px solid var(--border-strong)' };
        const confirmBase = { background: (variant === 'danger' ? 'var(--color-error-600)' : 'var(--color-primary-600)'), color: 'var(--text-inverse)' };
        const confirmHover = { background: (variant === 'danger' ? 'var(--color-error-700)' : 'var(--color-primary-700)') };

        cancelBtn?.addEventListener('click', () => cleanup(false));
        closeBtn?.addEventListener('click', () => cleanup(false));
        confirmBtn?.addEventListener('click', () => cleanup(true));

        setupHover(cancelBtn, cancelBase, cancelHover);
        setupHover(confirmBtn, confirmBase, confirmHover);
      });
    };
  }
} catch (_) {}

// Site-wide informational modal
// Usage: await window.showModal({ title, message, confirmText, variant })
// Returns a Promise<void>
try {
  if (!window.showModal) {
    window.showModal = function({ title = (window.I18N_UI?.notification || 'Notification'), message = '', confirmText = (window.I18N_UI?.okay || 'Okay'), variant = 'info' } = {}) {
      return new Promise(resolve => {
        const backdrop = document.createElement('div');
        backdrop.className = 'fixed inset-0 z-[9999] flex items-center justify-center p-4';
        backdrop.style.background = 'rgba(0,0,0,0.6)';
        backdrop.style.transition = 'opacity 180ms ease';
        backdrop.setAttribute('role', 'dialog');
        backdrop.setAttribute('aria-modal', 'true');
        backdrop.innerHTML = `
          <div class="rounded-2xl shadow-2xl max-w-md w-[92vw] p-6" style="background: var(--bg-card); color: var(--text-primary); border: 1px solid var(--border-default);">
            <div class="flex items-start justify-between">
              <h3 class="text-xl font-bold">${title}</h3>
              <button class="modal-close-btn" aria-label="${(window.I18N_UI?.close || 'Close')}" data-close style="color: var(--text-secondary);">
                <i class="fa-solid fa-xmark text-xl"></i>
              </button>
            </div>
            ${message ? `<p class="mt-3" style="color: var(--text-secondary);">${message}</p>` : ''}
            <div class="mt-5 flex gap-3 justify-end">
              <button class="rounded-lg font-semibold" data-confirm style="padding: 0.5rem 1rem; color: #fff; ${variant === 'danger' ? 'background: var(--color-error-600);' : 'background: var(--color-primary-600);'}">${confirmText}</button>
            </div>
          </div>
        `;

        function cleanup() {
          try {
            document.removeEventListener('keydown', onKeyDown);
            backdrop.removeEventListener('click', onBackdropClick);
          } catch (_) {}
          backdrop.style.opacity = '0';
          setTimeout(() => backdrop.remove(), 180);
          resolve();
        }

        function onBackdropClick(e) {
          if (e.target === backdrop) cleanup();
        }

        function onKeyDown(e) {
          if (e.key === 'Escape' || e.key === 'Enter') cleanup();
        }

        document.addEventListener('keydown', onKeyDown);
        backdrop.addEventListener('click', onBackdropClick);

        document.body.appendChild(backdrop);

        const confirmBtn = backdrop.querySelector('[data-confirm]');
        const closeBtn = backdrop.querySelector('[data-close]');

        function setupHover(el, base, hover) {
          if (!el) return;
          Object.assign(el.style, base);
          el.addEventListener('mouseenter', () => Object.assign(el.style, hover));
          el.addEventListener('mouseleave', () => Object.assign(el.style, base));
        }

        const confirmBase = { background: (variant === 'danger' ? 'var(--color-error-600)' : 'var(--color-primary-600)'), color: 'var(--text-inverse)' };
        const confirmHover = { background: (variant === 'danger' ? 'var(--color-error-700)' : 'var(--color-primary-700)') };

        closeBtn?.addEventListener('click', () => cleanup());
        confirmBtn?.addEventListener('click', () => cleanup());

        setupHover(confirmBtn, confirmBase, confirmHover);
      });
    };
  }
} catch (_) {}

document.addEventListener('DOMContentLoaded', function() {
    // Network status indicator
    const networkEl = document.getElementById('networkStatus');
    function setNetworkStatus(state) {
      if (!networkEl) return;
      networkEl.className = 'network-status ' + state;
      try {
        const t = window.I18N_NETWORK || {
          backOnline: 'Back online',
          offline: 'You are offline',
          reconnecting: 'Reconnecting…'
        };
        networkEl.textContent = state === 'online' ? t.backOnline : state === 'offline' ? t.offline : t.reconnecting;
      } catch (_) {
        networkEl.textContent = state === 'online' ? 'Back online' : state === 'offline' ? 'You are offline' : 'Reconnecting…';
      }
      networkEl.classList.remove('hidden');
      if (state === 'online') {
        setTimeout(() => networkEl.classList.add('hidden'), 1500);
      }
    }
    try {
      window.addEventListener('online', () => { setNetworkStatus('online'); });
      window.addEventListener('offline', () => { setNetworkStatus('offline'); });
    } catch (_) {}

    // Clear URL hash on page load
    if (window.location.hash) {
        history.replaceState(null, null, window.location.pathname + window.location.search);
    }
    
    // Removed demo video modal and handlers
    
    // Mobile Menu Toggle Functionality
    const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
    const mobileMenu = document.getElementById('mobile-menu');
    const hamburgerIcon = document.getElementById('hamburger-icon');
    const closeIcon = document.getElementById('close-icon');
    
    if (mobileMenuToggle && mobileMenu && hamburgerIcon && closeIcon) {
        // Ensure menu starts closed with class only (no inline CSS)
        mobileMenu.classList.add('hidden');
        hamburgerIcon.classList.remove('hidden');
        closeIcon.classList.add('hidden');
        mobileMenuToggle.setAttribute('aria-expanded', 'false');

        mobileMenuToggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();

            const isOpen = !mobileMenu.classList.contains('hidden');

            if (isOpen) {
                // Close menu
                mobileMenu.classList.add('hidden');
                hamburgerIcon.classList.remove('hidden');
                closeIcon.classList.add('hidden');
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
            } else {
                // Open menu
                mobileMenu.classList.remove('hidden');
                hamburgerIcon.classList.add('hidden');
                closeIcon.classList.remove('hidden');
                mobileMenuToggle.setAttribute('aria-expanded', 'true');
            }
        });

        // Close mobile menu when clicking on links
        const mobileNavLinks = document.querySelectorAll('.mobile-nav-link, .mobile-nav-button');
        mobileNavLinks.forEach(link => {
            link.addEventListener('click', function() {
                mobileMenu.classList.add('hidden');
                hamburgerIcon.classList.remove('hidden');
                closeIcon.classList.add('hidden');
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
            });
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', function(event) {
            const isClickInsideNav = mobileMenuToggle.contains(event.target) || mobileMenu.contains(event.target);
            if (!isClickInsideNav && !mobileMenu.classList.contains('hidden')) {
                mobileMenu.classList.add('hidden');
                hamburgerIcon.classList.remove('hidden');
                closeIcon.classList.add('hidden');
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
            }
        });

        // Close mobile menu on escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && !mobileMenu.classList.contains('hidden')) {
                mobileMenu.classList.add('hidden');
                hamburgerIcon.classList.remove('hidden');
                closeIcon.classList.add('hidden');
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
                mobileMenuToggle.focus();
            }
        });
    }
});

// Performance optimized intersection observer
document.addEventListener('DOMContentLoaded', function() {
  // Remove page loading class after DOM is ready
  const pageLoadingEl = document.querySelector('.page-loading');
  if (pageLoadingEl) {
    pageLoadingEl.classList.remove('page-loading');
  }
  
  // Add a slight delay to ensure smooth transition
  setTimeout(() => {
    // Make all loading-fade elements visible
    const loadingElements = document.querySelectorAll('.loading-fade');
    loadingElements.forEach((el, index) => {
      setTimeout(() => {
        el.classList.add('animate');
      }, index * 100); // Staggered animation
    });
    
    // Animate hero buttons with delay
    setTimeout(() => {
      const heroButtons = document.querySelectorAll('.hero-button');
      heroButtons.forEach((btn, index) => {
        setTimeout(() => {
          btn.classList.add('animate');
        }, index * 150);
      });
    }, 600); // After hero text animations
  }, 200);

  // Throttled scroll handler for better performance
  let scrollTimeout;
  let isScrolling = false;

  // Optimized intersection observer for animations
  const observerOptions = {
    threshold: [0.1, 0.5],
    rootMargin: '0px 0px -50px 0px'
  };

  const animationObserver = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting && entry.intersectionRatio >= 0.1) {
        // Add a small delay to prevent glitching
        setTimeout(() => {
          entry.target.classList.add('animate');
        }, 50);
        // Unobserve after animation to improve performance
        animationObserver.unobserve(entry.target);
      }
    });
  }, observerOptions);

  // Observe all animation elements
  const elements = document.querySelectorAll('.fade-in-up, .fade-in-left, .fade-in-right');
  elements.forEach(el => {
    animationObserver.observe(el);
  });

  // Enhanced navbar scroll effect with throttling
  function handleNavbarScroll() {
    const navbar = document.getElementById('navbar');
    if (!navbar) return;

    // Respect static navbar flag (e.g., landing page wants only sticky behavior)
    if (navbar.hasAttribute('data-static-navbar')) {
      navbar.classList.add('navbar-default');
      navbar.classList.remove('navbar-scrolled', 'navbar-floating');
      return;
    }

    if (isScrolling) return;

    isScrolling = true;
    requestAnimationFrame(() => {
      const nav = document.getElementById('navbar');
      if (!nav) { isScrolling = false; return; }

      const shouldScroll = window.scrollY > 100;

      // Preserve existing classes; only toggle state classes
      if (shouldScroll) {
        nav.classList.add('navbar-scrolled', 'navbar-floating');
        nav.classList.remove('navbar-default');
      } else {
        nav.classList.add('navbar-default');
        nav.classList.remove('navbar-scrolled', 'navbar-floating');
      }

      isScrolling = false;
    });
  }

  const navbarEl = document.getElementById('navbar');
  if (navbarEl) {
    // Initialize state on load
    handleNavbarScroll();
    // Attach scroll handler only if navbar is not static
    if (!navbarEl.hasAttribute('data-static-navbar')) {
      window.addEventListener('scroll', handleNavbarScroll, { passive: true });
    }
  }

  // Enhanced smooth scrolling with error handling
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const targetId = this.getAttribute('href');
      const target = document.querySelector(targetId);
      
      if (target) {
        // Check for reduced motion preference
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        
        target.scrollIntoView({
          behavior: prefersReducedMotion ? 'auto' : 'smooth',
          block: 'start'
        });
        
        // Update URL hash without triggering scroll
        if (targetId !== '#') {
          history.pushState(null, null, targetId);
        }
        
        // Focus management for accessibility
        if (target.hasAttribute('tabindex') || target.tagName === 'H1' || target.tagName === 'H2') {
          target.focus();
        }
      }
    });
  });







  // Enhanced CTA click tracking with error handling
  document.querySelectorAll('.cta-primary, .hero-button').forEach(button => {
    button.addEventListener('click', function(e) {
      try {
        // Visual feedback
        this.style.transform = 'scale(0.98)';
        setTimeout(() => {
          this.style.transform = '';
        }, 150);
        
        // Analytics tracking (if gtag is available)
        if (typeof gtag === 'function') {
          const label = this.textContent.trim() || 'CTA Click';
          gtag('event', 'click', {
            'event_category': 'CTA',
            'event_label': label,
            'value': 1
          });
        }
      } catch (error) {
        console.warn('CTA tracking error:', error);
      }
    });
  });

  // Removed demo video interaction tracking

  // Enhanced keyboard navigation
  document.addEventListener('keydown', function(e) {
    // ESC key to close modals/dropdowns
    if (e.key === 'Escape') {
      const openDropdowns = document.querySelectorAll('[aria-expanded="true"]');
      openDropdowns.forEach(dropdown => {
        dropdown.setAttribute('aria-expanded', 'false');
      });
    }
    
    // Tab trap for modal-like content
    if (e.key === 'Tab') {
      const focusableElements = document.querySelectorAll(
        'a[href], button, textarea, input[type="text"], input[type="radio"], input[type="checkbox"], select'
      );
      
      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];
      
      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    }
  });




  // Add main content landmark if not present
  const mainContent = document.querySelector('main, [role="main"]');
  if (!mainContent) {
    const heroSection = document.querySelector('header, .hero-section');
    if (heroSection) {
      heroSection.setAttribute('id', 'main-content');
      heroSection.setAttribute('role', 'main');
    }
  }

  // Announce page changes for screen readers
  const announcer = document.createElement('div');
  announcer.setAttribute('aria-live', 'polite');
  announcer.setAttribute('aria-atomic', 'true');
  announcer.className = 'sr-only';
  announcer.style.position = 'absolute';
  announcer.style.left = '-10000px';
  announcer.style.width = '1px';
  announcer.style.height = '1px';
  announcer.style.overflow = 'hidden';
  document.body.appendChild(announcer);

  // Function to announce messages to screen readers
  window.announceToScreenReader = function(message) {
    announcer.textContent = message;
    setTimeout(() => {
      announcer.textContent = '';
    }, 1000);
  };


});

  console.log('Eternalore: All scripts loaded successfully');