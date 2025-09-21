/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        // Use 'sans' for body text, 'serif' for headlines
        'sans': ['Source Sans 3', 'sans-serif'],
        'serif': ['Lora', 'serif'],
      },
      colors: {
        // Add The Economist's specific off-white and red
        'economist-red': '#E3120B', // You can use this or Tailwind's default red-800
        'economist-bg': '#F7F5F0',
      },
    },
  },
  plugins: [],
}