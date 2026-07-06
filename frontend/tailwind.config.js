/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#050914",
        card: "rgba(10, 15, 30, 0.6)",
        primary: "#00BFFF",
        secondary: "#00FFFF",
        muted: "#6B8CAE",
        border: "rgba(0, 191, 255, 0.15)",
        success: "#00E676",
        warning: "#FFB300",
        danger: "#FF4444",
        critical: "#FF0055",
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['Space Grotesk', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 4s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      }
    },
  },
  plugins: [],
}
