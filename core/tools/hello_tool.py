class HelloTool:
    name = 'hello'
    def run(self, params):
        who = params.get('who', 'world')
        return {'greeting': f'Hello, {who}!'}
