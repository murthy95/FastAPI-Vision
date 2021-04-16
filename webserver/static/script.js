// async function updatePredTable(preds) {
// 	const header = [ 'Label', 'Predictions %' ];

// 	let table = document.createElement('table');

// 	var tr = table.insertRow(-1);

// 	header.forEach((head) => {
// 		var th = document.createElement('th'); // TABLE HEADER.
// 		th.innerHTML = head;
// 		tr.appendChild(th);
// 	});

// 	preds.forEach((pred) => {
// 		tr = table.insertRow(-1);

// 		var tabCell = tr.insertCell(-1);
// 		tabCell.innerHTML = pred['label'];

// 		var predLabel = tr.insertCell(-1);
// 		predLabel.innerHTML = Math.round(pred['probability'] * 10000) / 100;
// 	});

// 	const divContainer = document.getElementById('pred-table');
// 	divContainer.innerHTML = '';
// 	divContainer.appendChild(table);
// }

async function updatePredTable(preds) {
	var img = document.getElementById('predimg');
	img.src = "data:image/png;base64, " + preds['result'];
	console.log(preds["result"])
	img.width = preds["width"];
	img.height = preds["height"];
	img.alt = "output";
	document.body.appendChild(img);
}

async function addImage(file) {
	if (img.src !== URL.createObjectURL(file)) {
		let imgDiv = document.getElementById('img-div');
		imgDiv.innerHTML = '';

		var img = document.createElement('img');
		img.src = URL.createObjectURL(file);
		imgDiv.append(img);
	}
}

async function postResult(file) {
	const formData = new FormData();
	formData.append('img_file', file);
	let results = await axios
		.post(`/predict`, formData, {
			headers: {
				'Content-Type': 'multipart/form-data'
			}
		})
		.catch((error) => {
			console.log('bad');
		});

	return results['data']['predictions'];
}

document.getElementById('btn').addEventListener('click', async function () {
	let file = document.getElementById('file-input').files[0];

	addImage(file);

	const preds = await postResult(file);

	updatePredTable(preds);
});
