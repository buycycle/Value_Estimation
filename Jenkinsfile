def ab = ""
def image_tag = ""
def environment = ""
def environments_list = [
    "staging": "staging",
    "main": "live"
]
def map_branch_to_ab = [
    "dev": "canary",
    "staging": "canary",
    "main": "stable"
]

if (env.BRANCH_NAME == "main" || env.BRANCH_NAME == "staging") {
    ab = map_branch_to_ab[env.BRANCH_NAME]
    environment = environments_list[env.BRANCH_NAME]
    image_tag = "${environments_list[env.BRANCH_NAME]}-${env.BUILD_NUMBER}"
}
else {
	ab = "canary"
    environment = "dev"
    image_tag = "dev-${env.BUILD_NUMBER}"
}

pipeline {
    agent any
    options {
        disableConcurrentBuilds()
    }
    stages {
        stage('Build') {
            steps {
                sh "aws s3 cp s3://buycycle-env-bucket/rec-api-temp/config.ini config/config.ini"
                script{
                    if (environment == "live") {
                        app = docker.build("price", "-f docker/main.dockerfile --build-arg ENVIRONMENT=${environment} --build-arg AB=${ab} .")
                    }
                    else {
                        app = docker.build("price", "-f docker/dev.dockerfile --build-arg ENVIRONMENT=${environment} --build-arg AB=${ab} .")
                    }

                    test = docker.build("price-test", "-f docker/test.dockerfile .")

                }
            }
        }
        stage('Test'){
            steps {
                script{
                    test.inside{
                            sh 'mkdir -p data'
                            sh 'python -m unittest discover'
                            sh 'pytest -v tests/'
                        }
                    }
                }
        }
        stage('Push Docker image') {
            steps {
                script{
                    docker.withRegistry('https://930985312118.dkr.ecr.eu-central-1.amazonaws.com/price', 'ecr:eu-central-1:aws-credentials-ecr') {
                        app.push(image_tag)
                        app.push("latest")
                    }
                }
            }
        }
        stage("Modify HELM chart") {
           steps {
                sh "make push IMAGE_TAG=${image_tag} ENV=${environment}"
            }
        }
        stage("Sync Chart") {
           steps {
            withCredentials([string(credentialsId: 'argocd-token', variable: 'TOKEN')]) {
                script {
                   env.namespace = environment
                }
                sh '''
                  set +x
                  argocd app sync price-$namespace --server argocd.cube-gebraucht.com --auth-token $TOKEN
                  argocd app wait price-$namespace --server argocd.cube-gebraucht.com --auth-token $TOKEN
                '''
              }
            }
        }
    }
}
