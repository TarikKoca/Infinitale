// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      fontFamily: {
        inter: ['Inter', 'sans-serif'],
      },
      colors: {
        // Primary palette driven by CSS variables for runtime theming
        primary: {
          50: 'var(--color-primary-50)',
          100: 'var(--color-primary-100)',
          200: 'var(--color-primary-200)',
          300: 'var(--color-primary-300)',
          400: 'var(--color-primary-400)',
          500: 'var(--color-primary-500)',
          600: 'var(--color-primary-600)',
          700: 'var(--color-primary-700)',
          800: 'var(--color-primary-800)',
          900: 'var(--color-primary-900)'
        },
        success: {
          50: 'var(--color-success-50)',
          100: 'var(--color-success-100)',
          600: 'var(--color-success-600)',
          700: 'var(--color-success-700)'
        },
        warning: {
          50: 'var(--color-warning-50)',
          100: 'var(--color-warning-100)',
          600: 'var(--color-warning-600)',
          700: 'var(--color-warning-700)'
        },
        error: {
          50: 'var(--color-error-50)',
          100: 'var(--color-error-100)',
          600: 'var(--color-error-600)',
          700: 'var(--color-error-700)'
        },
        surface: {
          primary: 'var(--bg-primary)',
          secondary: 'var(--bg-secondary)',
          card: 'var(--bg-card)'
        },
        text: {
          primary: 'var(--text-primary)',
          secondary: 'var(--text-secondary)',
          muted: 'var(--text-muted)'
        },
        border: {
          DEFAULT: 'var(--border-default)',
          light: 'var(--border-light)',
          medium: 'var(--border-medium)',
          strong: 'var(--border-strong)'
        }
      }
    },
  },
};

