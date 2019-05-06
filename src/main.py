from flask import Flask
from flask_restplus import Resource, Api, reqparse
from flask_cors import CORS

app = Flask(__name__)
app.config['BUNDLE_ERRORS'] = True
api = Api(app, version="1.0", title="Back-end")
CORS(app)


data = reqparse.RequestParser()
data.add_argument('email', type=str, required=True, location='args')
data.add_argument('psw', type=str, required=True, location='args')


@api.route('/signIn',  endpoint='with-parser')
class SignIn(Resource):
    @api.expect(data)
    def get(self):
        args = data.parse_args(strict=True)
        print(args['email'] + " " + args['psw'])
        return {'token': 'token'}


if __name__ == '__main__':
    app.run(debug=True)
