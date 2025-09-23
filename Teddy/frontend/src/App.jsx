import { useState } from "react";
import { CallPython } from "../wailsjs/go/main/App";

function App() {
  const [msg, setMsg] = useState("");

  async function greetScript() {
    const res = await CallPython("Deepansh");
    setMsg(res);
  }

  async function greetAPI() {
    const res = await fetch("http://127.0.0.1:8000/greet/Deepansh");
    const data = await res.json();
    setMsg(data.message);
  }

  return (
    <div>
      <h1>{msg || "Click a button"}</h1>
      <button onClick={greetScript}>Greet via Python Script</button>
      <button onClick={greetAPI}>Greet via FastAPI</button>
    </div>
  );
}

export default App;
