import { useEffect } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Layout } from "./components/Chrome";
import { SessionProvider } from "./state";
import Dashboard from "./pages/Dashboard";
import Intake from "./pages/Intake";
import Review from "./pages/Review";
import Result from "./pages/Result";

/**
 * Reset scroll on navigation.
 *
 * Without this, confirming from a scrolled-down review table lands the user partway into
 * the result page — below the decision headline, which is the one thing they must see.
 */
function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

export default function App() {
  return (
    <SessionProvider>
      <ScrollToTop />
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/:disease" element={<Intake />} />
          <Route path="/:disease/review" element={<Review />} />
          <Route path="/:disease/result" element={<Result />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </SessionProvider>
  );
}
