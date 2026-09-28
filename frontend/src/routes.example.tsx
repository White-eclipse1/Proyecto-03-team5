/**
 * Este archivo es solo un EJEMPLO de integración — no reemplaza tu router
 * existente. Agrega estas dos rutas donde ya tengas la de Search.
 */
import { Route, Routes } from "react-router-dom";
import { AnnotateScreen } from "./components/annotate/AnnotateScreen";
import { UploadScreen } from "./components/upload/UploadScreen";
// import { SearchScreen } from "./components/search/SearchScreen"; // ya existente

export function ExampleRoutes() {
  return (
    <Routes>
      {/* <Route path="/search" element={<SearchScreen />} /> */}
      <Route path="/upload" element={<UploadScreen />} />
      <Route path="/annotate/:imageId" element={<AnnotateScreen />} />
    </Routes>
  );
}
