import React from 'react';
import {Box, Text} from 'ink';

const VERSION = '0.1.0';
const BUILT_BY = 'Aditya';

// NIA caduceus (ported from banner.py CADUCEUS)
// Uses braille art for a cleaner look. Colors: gold (#FFD700) → orange (#FFBF00) → bronze (#CD7F32) → dark gold (#B8860B)
const CADUCEUS_LINES: {text: string; color: string}[] = [
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#CD7F32'},
	{text: '⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀', color: '#CD7F32'},
	{text: '⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀', color: '#FFBF00'},
	{text: '⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀', color: '#FFBF00'},
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#FFD700'},
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#FFD700'},
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#FFBF00'},
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#FFBF00'},
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#CD7F32'},
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#CD7F32'},
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#B8860B'},
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#B8860B'},
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#B8860B'},
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#B8860B'},
	{text: '⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀', color: '#B8860B'},
];

// NIA in block letters (same style as NIA logo)
const NIA_LOGO: {text: string; color: string}[] = [
	{text: '██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗', color: '#FFD700'},
	{text: '██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝', color: '#FFD700'},
	{text: '███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗', color: '#FFBF00'},
	{text: '██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║', color: '#FFBF00'},
	{text: '██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║', color: '#CD7F32'},
	{text: '╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝', color: '#CD7F32'},
];

interface WelcomeBannerProps {
	model?: string;
	provider?: string;
	sessionId?: string;
	cwd?: string;
	toolCount?: number;
	skillCount?: number;
	tools?: string[];
	skills?: {name: string; category: string}[];
}

export function WelcomeBanner({
	model,
	provider,
	sessionId,
	cwd,
	toolCount = 0,
	skillCount = 0,
	tools = [],
	skills = [],
}: WelcomeBannerProps): React.JSX.Element {
	// Group skills by category (like NIA does)
	const skillsByCategory: Record<string, string[]> = {};
	for (const skill of skills.slice(0, 30)) {
		if (!skillsByCategory[skill.category]) {
			skillsByCategory[skill.category] = [];
		}
		skillsByCategory[skill.category].push(skill.name);
	}

	return (
		<Box flexDirection="column" marginBottom={1}>
			{/* Two-column layout: caduceus on left, NIA + info on right */}
			<Box flexDirection="row">
				{/* Left column: caduceus */}
				<Box flexDirection="column" marginRight={3}>
					{CADUCEUS_LINES.map((line, i) => (
						<Text key={`cad-${i}`} color={line.color}>{line.text}</Text>
					))}
				</Box>

				{/* Right column: NIA name + model + built by + tools/skills */}
				<Box flexDirection="column" flexGrow={1}>
					{/* NIA block letters */}
					{NIA_LOGO.map((line, i) => (
						<Text key={`nia-${i}`} color={line.color} bold>{line.text}</Text>
					))}

					<Text> </Text>

					{/* Model + provider */}
					{model ? (
						<Box>
							<Text color="#FFD700" bold>{model.split('/').pop() || model}</Text>
							{provider ? <Text dimColor> · {provider}</Text> : null}
						</Box>
					) : null}

					{/* Built by */}
					<Box>
						<Text dimColor>Built by </Text>
						<Text color="#FF8C00" bold>{BUILT_BY}</Text>
						<Text dimColor> · v{VERSION}</Text>
					</Box>

					<Text> </Text>

					{/* Tools and skills summary */}
					<Box>
						<Text color="#FF8C00" bold>{toolCount}</Text>
						<Text dimColor> tools</Text>
						<Text dimColor> · </Text>
						<Text color="#BD93F9" bold>{skillCount}</Text>
						<Text dimColor> skills</Text>
					</Box>

					{/* Session + path */}
					{sessionId ? (
						<Box>
							<Text dimColor>Session: {sessionId}</Text>
						</Box>
					) : null}
					{cwd ? (
						<Box>
							<Text dimColor>{cwd}</Text>
						</Box>
					) : null}
				</Box>
			</Box>

			<Text> </Text>

			{/* Skills list (if available) */}
			{Object.keys(skillsByCategory).length > 0 ? (
				<Box flexDirection="column">
					<Text color="#FFD700" bold>Available Skills</Text>
					{Object.entries(skillsByCategory).slice(0, 8).map(([cat, names]) => (
						<Box key={cat}>
							<Text color="#BD93F9">{cat}: </Text>
							<Text dimColor>{names.join(', ')}</Text>
						</Box>
					))}
					{Object.keys(skillsByCategory).length > 8 ? (
						<Text dimColor>  ...and {Object.keys(skillsByCategory).length - 8} more categories</Text>
					) : null}
				</Box>
			) : null}

			<Text> </Text>

			{/* Command hints */}
			<Box>
				<Text dimColor>
					{'  '}/help for commands · /tools to list · /skills to browse · /model to switch · Ctrl+C exit
				</Text>
			</Box>
			<Text> </Text>
		</Box>
	);
}
