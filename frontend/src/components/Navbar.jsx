import { NavLink } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <NavLink to="/" className="navbar-logo">
          <div className="logo-mark">
            <svg viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect x="2" y="3" width="12" height="1.5" rx="0.75" fill="white"/>
              <rect x="2" y="7" width="8" height="1.5" rx="0.75" fill="white"/>
              <rect x="2" y="11" width="10" height="1.5" rx="0.75" fill="white"/>
            </svg>
          </div>
          <span className="logo-wordmark">SLANT</span>
          <span className="logo-version">v2</span>
        </NavLink>

        <ul className="navbar-links">
          <li><NavLink to="/" end>Analyze</NavLink></li>
          <li><NavLink to="/history">History</NavLink></li>
          <li><NavLink to="/engines">How it Works</NavLink></li>
          <li><NavLink to="/about">About</NavLink></li>
        </ul>
      </div>
    </nav>
  );
}
