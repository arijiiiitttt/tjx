import '@fontsource-variable/geist';
import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import { App } from './app/App';

createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
