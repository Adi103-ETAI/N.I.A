import React from 'react';
import {render} from 'ink';
import * as fs from 'node:fs';

import {App} from './App.js';
import type {FrontendConfig} from './types.js';

function loadConfig(): FrontendConfig {
	// Try env var first (primary path).
	const envConfig = process.env.NIAHARNESS_FRONTEND_CONFIG;
	if (envConfig) {
		try {
			return JSON.parse(envConfig) as FrontendConfig;
		} catch {}
	}

	// Fallback: read from temp file (set by react_launcher.py).
	const configFile = process.env.NIAHARNESS_FRONTEND_CONFIG_FILE;
	if (configFile) {
		try {
			return JSON.parse(fs.readFileSync(configFile, 'utf-8')) as FrontendConfig;
		} catch {}
	}

	// Last resort: empty config (will show an error in the UI).
	return {backend_command: []} as FrontendConfig;
}

const config = loadConfig();

render(<App config={config} />);
