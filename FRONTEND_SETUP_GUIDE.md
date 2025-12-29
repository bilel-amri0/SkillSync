# SkillSync Frontend Setup Guide

## Phase 2: Frontend MVP - Complete Setup Instructions

### Step 1: Create Vite Project

```cmd
cd c:\Users\Lenovo\Downloads\SkillSync_Enhanced
npm create vite@latest frontend -- --template react-ts
```

When prompted, select:
- Framework: **React**
- Variant: **TypeScript**

### Step 2: Navigate and Install Dependencies

```cmd
cd frontend
npm install
npm install axios lucide-react framer-motion clsx
npm install -D tailwindcss postcss autoprefixer
```

### Step 3: Initialize Tailwind CSS

```cmd
npx tailwindcss init -p
```

### Step 4: Configure Tailwind

Replace the content of `tailwind.config.js`:

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

### Step 5: Update CSS

Replace `src/index.css` with Tailwind directives (code provided separately).

### Step 6: Add API Service

Create `src/services/api.ts` (code provided separately).

### Step 7: Create Components

Create these files:
- `src/components/CVUploader.tsx`
- `src/components/TemplateSelector.tsx`
- `src/components/ColorSchemeSelector.tsx`
- `src/components/LoadingSpinner.tsx`

### Step 8: Update App.tsx

Replace `src/App.tsx` with the main application logic (code provided separately).

### Step 9: Run the Frontend

```cmd
npm run dev
```

The frontend will be available at: **http://localhost:5173**

### Step 10: Test End-to-End

1. Make sure backend is running: `http://localhost:8000`
2. Open frontend: `http://localhost:5173`
3. Upload a CV file
4. Select a template and color scheme
5. Click "Download Portfolio"
6. ZIP file should download automatically

---

## Quick Start Command Summary

```cmd
# In SkillSync_Enhanced directory
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install axios lucide-react framer-motion clsx
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Then replace files with provided code
# Finally:
npm run dev
```

---

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── CVUploader.tsx
│   │   ├── TemplateSelector.tsx
│   │   ├── ColorSchemeSelector.tsx
│   │   └── LoadingSpinner.tsx
│   ├── services/
│   │   └── api.ts
│   ├── App.tsx
│   ├── index.css
│   └── main.tsx
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

---

## Troubleshooting

**Port already in use:**
```cmd
# Change port in vite.config.ts or kill process on 5173
```

**CORS errors:**
- Backend CORS is already configured to allow localhost:5173
- If issues persist, restart backend

**TypeScript errors:**
```cmd
npm install --save-dev @types/node
```
