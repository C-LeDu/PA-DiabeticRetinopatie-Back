from flask_restplus import Resource, reqparse


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
