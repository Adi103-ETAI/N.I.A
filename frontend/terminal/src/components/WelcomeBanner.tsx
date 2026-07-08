import React from 'react';
import {Box, Text} from 'ink';

const VERSION = '0.1.0';
const BUILT_BY = 'Adi103-ETAI';

// Caduceus ASCII art (medical symbol — NIA wanted to be a doctor)
const CADUCEUS = [
	'                    .:.',
	"                   .' '.",
	'              .-.-.`     `.-.-.',
	"            .'     \\       /     `.",
	'                   _\\     /_',
	'              .-"".__\\   /__.""-.',
	'             /    .-"  |  "-.    \\',
	'            |   .\'  .-"|`"-.  `.   |',
	'            |  |   |   |   |   |  |',
	'            |  |   |.-.|.-.|   |  |',
	'            |  |   |   |   |   |  |',
	'            |  |   |.-.|.-.|   |  |',
	'            |  |   |   |   |   |  |',
	'            |  |   |.-.|.-.|   |  |',
	'            |  |   |   |   |   |  |',
	"            \\  `. |.-.|.-.| .'  /",
	"            `-._|____|____|_.-'",
	'                |    |    |',
	'                |    |    |',
	'               /|    |    |\\',
	'              / |    |    | \\',
	'                |____|____|',
	'                \\    |    /',
	'                 \\   |   /',
	'                  \\  |  /',
	'                   \\ | /',
	'                    \\|/',
	"                    ` '",
];

// NIA in block letters
const NIA_NAME = [
	'  ███╗   ██╗ █████╗ ███████╗',
	'  ████╗  ██║██╔══██╗██╔════╝',
	'  ██╔██╗ ██║███████║███████╗',
	'  ██║╚██╗██║██╔══██║╚════██║',
	'  ██║ ╚████║██║  ██║███████║',
	'  ╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝',
];

interface WelcomeBannerProps {
	model?: string;
	provider?: string;
	sessionId?: string;
	cwd?: string;
	toolCount?: number;
	skillCount?: number;
}

export function WelcomeBanner({
	model,
	provider,
	sessionId,
	cwd,
	toolCount = 0,
	skillCount = 0,
}: WelcomeBannerProps): React.JSX.Element {
	return (
		<Box flexDirection="column" marginBottom={1}>
			{/* Caduceus + NIA name */}
			<Box flexDirection="column" alignItems="center">
				{CADUCEUS.map((line, i) => (
					<Text key={`cad-${i}`} color="#FFD700">{line}</Text>
				))}
				<Text> </Text>
				{NIA_NAME.map((line, i) => (
					<Text key={`nia-${i}`} color="#FFD700" bold>{line}</Text>
				))}
				<Text> </Text>
				<Text color="#00CED1" bold>Neural Intelligence Assistant</Text>
				<Text dimColor>Your AI partner, inspired by J.A.R.V.I.S</Text>
			</Box>

			<Text> </Text>
			<Text color="#B8860B">{'─'.repeat(72)}</Text>
			<Text> </Text>

			{/* Session info panel */}
			<Box flexDirection="column" borderStyle="round" borderColor="#B8860B" paddingX={2} paddingY={0}>
				<Box>
					<Text color="#6272A4"> Model:     </Text>
					<Text color="#00CED1">{model || 'unknown'}</Text>
					{provider ? <Text dimColor> · {provider}</Text> : null}
				</Box>
				{sessionId ? (
					<Box>
						<Text color="#6272A4"> Session:   </Text>
						<Text dimColor>{sessionId}</Text>
					</Box>
				) : null}
				{cwd ? (
					<Box>
						<Text color="#6272A4"> Path:      </Text>
						<Text dimColor>{cwd}</Text>
					</Box>
				) : null}
				<Box>
					<Text color="#6272A4"> Built by:  </Text>
					<Text color="#FF8C00" bold>{BUILT_BY}</Text>
					<Text dimColor>{'  '}v{VERSION}</Text>
				</Box>
			</Box>

			<Text> </Text>

			{/* Tools and skills panel */}
			<Box flexDirection="column" borderStyle="round" borderColor="#B8860B" paddingX={2} paddingY={0}>
				<Box>
					<Text color="#FF8C00" bold>  {toolCount}</Text>
					<Text dimColor> tools available</Text>
					<Text>    </Text>
					<Text color="#BD93F9" bold>{skillCount}</Text>
					<Text dimColor> skills loaded</Text>
				</Box>
				<Text> </Text>
				<Text dimColor>
					{'  '}/help for commands · /tools to list · /skills to browse · /model to switch
				</Text>
			</Box>

			<Text> </Text>
			<Text dimColor>  Type your message below. Press Enter to send.</Text>
			<Text> </Text>
		</Box>
	);
}
