import React from "react";
import { createRoot } from "react-dom/client";
import "@dmitrymake/rk-ui/styles.css";
import "./article.css";
import App from "./App.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
