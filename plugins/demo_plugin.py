class DemoPlugin:
    name = 'demo_plugin'
    def run(self, params):
        return {'message': 'Hello from the demo plugin!', 'params': params}
