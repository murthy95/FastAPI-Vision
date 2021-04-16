from locust import HttpUser, TaskSet, task


class WebsiteUser(HttpUser):
    min_wait = 500
    max_wait = 5000

    @task
    def predict(self):
        with open('American_Eskimo_Dog_1.jpg', 'rb') as image:
            self.client.post('/predict', files={'img_file': image})
