import '@testing-library/jest-dom/vitest';
import {
  cleanup,
  render,
  screen,
} from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import {
  afterEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest';

import { DashboardPage } from '../src/pages/Dashboard';

vi.mock('../src/hooks/useDashboardSummary', () => ({
  useDashboardSummary: () => ({
    status: 'loading',
  }),
}));

afterEach(() => {
  cleanup();
});

function renderDashboard() {
  render(
    <MemoryRouter>
      <DashboardPage />
    </MemoryRouter>,
  );
}

describe('SPEC-COCO-UI-001 - exportación COCO desde Dashboard', () => {
  it('muestra la acción Exportar COCO', () => {
    renderDashboard();

    expect(
      screen.getByRole('link', {
        name: /exportar coco/i,
      }),
    ).toBeInTheDocument();
  });

  it('la acción apunta al endpoint de exportación COCO', () => {
    renderDashboard();

    const exportLink = screen.getByRole('link', {
      name: /exportar coco/i,
    });

    expect(exportLink).toHaveAttribute(
      'href',
      '/api/export/coco',
    );
  });
});