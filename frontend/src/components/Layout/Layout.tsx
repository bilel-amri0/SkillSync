import { useState } from 'react';
import type { ReactNode } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { 
  LayoutDashboard, 
  FileText, 
  Briefcase, 
  Brain, 
  Target,
  Menu,
  X,
  LogOut,
  User
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { getStoredUser } from '../../services/authService';

interface LayoutProps {
  children?: ReactNode;
  onLogout: () => void;
}

// Get initial user data
const getInitialUserData = () => {
  const user = getStoredUser();
  return {
    name: user?.full_name || user?.username || 'User',
    email: user?.email || ''
  };
};

const Layout = ({ children, onLogout }: LayoutProps) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const initialUser = getInitialUserData();
  const [userName] = useState(initialUser.name);
  const [userEmail] = useState(initialUser.email);
  const navigate = useNavigate();
  const location = useLocation();

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
    { name: 'CV Analysis', href: '/cv-analysis', icon: FileText },
    { name: 'Career Guidance', href: '/career-guidance', icon: Brain },
    { name: 'Job Matching', href: '/job-matching', icon: Briefcase },
    { name: 'Interview Practice', href: '/interview', icon: Target },
  ];

  const handleNavigation = (href: string) => {
    navigate(href);
    setSidebarOpen(false);
  };

  const handleLogout = () => {
    onLogout();
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-gray-600 bg-opacity-75 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Mobile sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 shadow-lg lg:hidden"
          >
            <div className="flex items-center justify-between px-4 py-4">
              <div className="flex items-center">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-600 text-white font-bold">
                  SS
                </div>
                <span className="ml-3 text-lg font-semibold text-gray-900 dark:text-white">SkillSync</span>
              </div>
              <button
                onClick={() => setSidebarOpen(false)}
                className="rounded-md p-2 text-gray-400 hover:text-gray-500"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
            <nav className="space-y-1 px-4 mt-6">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href;
                return (
                  <button
                    key={item.name}
                    onClick={() => handleNavigation(item.href)}
                    className={`group flex w-full items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                      isActive
                        ? 'bg-blue-50 text-blue-700 dark:bg-blue-900 dark:text-blue-200'
                        : 'text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700'
                    }`}
                  >
                    <item.icon className="mr-3 h-5 w-5" />
                    {item.name}
                  </button>
                );
              })}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col">
        <div className="flex min-h-0 flex-1 flex-col bg-white dark:bg-gray-800 shadow-lg">
          <div className="flex h-16 flex-shrink-0 items-center px-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-600 text-white font-bold">
                SS
              </div>
              <div className="ml-3">
                <p className="text-lg font-semibold text-gray-900 dark:text-white">SkillSync</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">AI Career Hub</p>
              </div>
            </div>
          </div>
          <div className="flex flex-1 flex-col overflow-y-auto">
            <nav className="flex-1 space-y-1 px-4 py-4">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href;
                return (
                  <button
                    key={item.name}
                    onClick={() => handleNavigation(item.href)}
                    className={`group flex w-full items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                      isActive
                        ? 'bg-blue-50 text-blue-700 dark:bg-blue-900 dark:text-blue-200'
                        : 'text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700'
                    }`}
                  >
                    <item.icon className="mr-3 h-5 w-5" />
                    {item.name}
                  </button>
                );
              })}
            </nav>
            <div className="border-t border-gray-200 dark:border-gray-700 p-4">
              <button
                onClick={handleLogout}
                className="group flex w-full items-center px-3 py-2 text-sm font-medium text-gray-700 rounded-lg hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700 transition-colors"
              >
                <LogOut className="mr-3 h-5 w-5" />
                Sign out
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top header */}
        <div className="sticky top-0 z-40 flex h-16 items-center gap-x-4 border-b border-gray-200 bg-white px-4 shadow-sm dark:border-gray-700 dark:bg-gray-800 sm:gap-x-6 sm:px-6 lg:px-8">
          <button
            type="button"
            className="-m-2.5 p-2.5 text-gray-700 lg:hidden dark:text-gray-300"
            onClick={() => setSidebarOpen(true)}
          >
            <span className="sr-only">Open sidebar</span>
            <Menu className="h-6 w-6" aria-hidden="true" />
          </button>

          <div className="h-6 w-px bg-gray-200 dark:bg-gray-700 lg:hidden" aria-hidden="true" />

          <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
            <div className="relative flex flex-1">
              <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                <svg className="h-5 w-5 text-gray-400" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                  <path
                    fillRule="evenodd"
                    d="M9 3a6 6 0 104.472 10.03l4.249 4.249a1 1 0 001.414-1.414l-4.249-4.249A6 6 0 009 3zM5 9a4 4 0 118 0 4 4 0 01-8 0z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <input
                type="search"
                placeholder="Search platform..."
                className="block w-full rounded-md border-0 py-1.5 pl-10 pr-3 text-gray-900 placeholder:text-gray-400 focus:ring-2 focus:ring-blue-600 sm:text-sm dark:bg-gray-700 dark:text-gray-100"
              />
            </div>
            <div className="flex items-center gap-x-4 lg:gap-x-6">
              <div className="relative">
                <span className="inline-flex items-center rounded-md bg-green-50 px-2 py-1 text-xs font-medium text-green-700 dark:bg-green-900 dark:text-green-200">
                  Online
                </span>
              </div>
              <div className="hidden lg:block lg:h-6 lg:w-px lg:bg-gray-200 lg:dark:bg-gray-700" aria-hidden="true" />
              <div className="flex items-center">
                <div className="mr-3 text-right">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">{userName}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">{userEmail || 'Premium Plan'}</p>
                </div>
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300">
                  <User className="h-5 w-5" />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="py-6">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            {children || <Outlet />}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;
