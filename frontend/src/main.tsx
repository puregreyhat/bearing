import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import App from "./App"
import AdminPanel from "./AdminPanel"
import "./index.css"

const Root = window.location.pathname === "/admin" ? AdminPanel : App

createRoot(document.getElementById("app")!).render(
  <StrictMode><Root /></StrictMode>,
)
