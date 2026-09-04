import { useEffect } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AuthProvider } from "./auth";
import { Layout } from "./components/Chrome";
import { RequireAuth } from "./components/RequireAuth";
import { SessionProvider } from "./state";
import About from "./pages/About";
import Account from "./pages/Account";
import Dashboard from "./pages/Dashboard";
import Forgot from "./pages/Forgot";
import Intake from "./pages/Intake";
import Landing from "./pages/Landing";
import Reset from "./pages/Reset";
import Result from "./pages/Result";
import Review from "./pages/Review";
import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";
import Start from "./pages/Start";

/**
 * Reset scroll on navigation.
 *
 * Without this, confirming from a scrolled-down review table lands the user partway into
 * the result page — below the headline, which is the one thing they must see.
 */
function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

/**
 * Everything under `/app` shares one gate and one frame.
 *
 * The gate is a single wrapper rather than a check inside each page: a page that forgot to
 * check would render a form for health data to someone with no account, and the failure
 * would be invisible until someone looked. Here, an unprotected route has to be written
 * outside this element on purpose.
 */
function ProtectedArea() {
  return (
    <RequireAuth>
      <Layout>
        <Routes>
          <Route index element={<Dashboard />} />
          <Route path="account" element={<Account />} />
          {/* Upload is the primary path; the guided form is reached from it. */}
          <Route path=":disease" element={<Start />} />
          <Route path=":disease/enter" element={<Intake />} />
          <Route path=":disease/review" element={<Review />} />
          <Route path=":disease/result" element={<Result />} />
          <Route path=":disease/about" element={<About />} />
          <Route path="*" element={<Navigate to="/app" replace />} />
        </Routes>
      </Layout>
    </RequireAuth>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <SessionProvider>
        <ScrollToTop />
        <Routes>
          {/* Public — and this list is the whole of it. */}
          <Route path="/" element={<Landing />} />
          <Route path="/signin" element={<SignIn />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/forgot" element={<Forgot />} />
          <Route path="/reset/:token" element={<Reset />} />

          <Route path="/app/*" element={<ProtectedArea />} />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </SessionProvider>
    </AuthProvider>
  );
}
